import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import argparse
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
import pdb

from einops import rearrange, repeat
import einops
from models.vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d
from enum import Enum

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x

class TensorTrans(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
    def forward(self, x: torch.Tensor):
        return x.permute(0, 3, 1, 2)


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=1,
            qkv_block_size=4,
            proj_bias=False,
            conv_bias=True,
            kernel_size=4,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.kernel_size = kernel_size

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.conv1d = CausalConv1d(
            dim=inner_dim,
            kernel_size=kernel_size,
            bias=conv_bias,
        )
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()
    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv1d(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class xLSTMLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        self.fc2 = ViLLayer(dim=hidden_features, direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT)
        self.fc3 = ViLLayer(dim=hidden_features, direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc2(x)
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x)
        x = self.dwconv_3(x, H, W)

    
        return x

class xLSTMBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm1 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = xLSTMLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

        self.multi_scale_5x5 = Pooling(5)
        self.multi_scale_9x9 = Pooling(9)
        self.multi_scale_13x13 = Pooling(13)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x_norm = self.norm1(x)
        x_pooled = self.multi_scale_5x5(x_norm) + self.multi_scale_9x9(x_norm) + self.multi_scale_13x13(x_norm)
        x = x + x_pooled
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class Model(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes=1, input_channels=3, img_size=224, embed_dims=[32, 64, 128, 256, 512], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(input_channels, embed_dims[0], 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(embed_dims[0], embed_dims[1], 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(embed_dims[1], embed_dims[2], 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(embed_dims[0])
        self.ebn2 = nn.BatchNorm2d(embed_dims[1])
        self.ebn3 = nn.BatchNorm2d(embed_dims[2])
        
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])

        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([xLSTMBlock(
            dim=embed_dims[3], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([xLSTMBlock(
            dim=embed_dims[4],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([xLSTMBlock(
            dim=embed_dims[3], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([xLSTMBlock(
            dim=embed_dims[2], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[3],
                                              embed_dim=embed_dims[4])

        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(embed_dims[3], embed_dims[2], 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(embed_dims[2], embed_dims[1], 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(embed_dims[1], embed_dims[0], 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(embed_dims[0], embed_dims[0]//2, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(embed_dims[3])
        self.dbn2 = nn.BatchNorm2d(embed_dims[2])
        self.dbn3 = nn.BatchNorm2d(embed_dims[1])
        self.dbn4 = nn.BatchNorm2d(embed_dims[0])
        
        self.final = nn.Conv2d(embed_dims[0]//2, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)
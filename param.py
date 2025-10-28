import tmunet
from thop import profile
import torch

img_size = 256
model_size = 't'

if model_size == 'b':
    dim_group = [32, 64, 128, 256, 512]
if model_size == 's':
    dim_group = [16, 32, 64, 128, 256]
if model_size == 't':
    dim_group = [8, 16, 32, 64, 128]
model = ulite_xlstm.Model(num_classes=1, img_size=img_size, embed_dims=dim_group)

randn_input = torch.randn(1, 3, img_size, img_size) 
flops, params = profile(model, inputs=(randn_input, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')

trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(trainable_num/1000**2)

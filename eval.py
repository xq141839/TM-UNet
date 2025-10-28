import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from dataloader import BinaryLoader
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import tmunet
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image
import json
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance, compute_percent_hausdorff_distance, HausdorffDistanceMetric

def hd_score(p, y):

    tmp_hd = compute_hausdorff_distance(p, y)
    tmp_hd = torch.mean(tmp_hd)

    return tmp_hd.item()

class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

def get_transform():
   return A.Compose(
       [
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='MoNuSeg', help='CVCCDB, MoNuSeg, Ultrasound, ColonDB')
    parser.add_argument('--model_name', type=str,default='ulite_xlstm', help='')
    parser.add_argument('--model_size', type=str,default='t', help='')
    parser.add_argument('--img_size', type=int,default=512, help='')
    parser.add_argument('--checkpoints',default='outputs/xxxxxxx.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    pred_path = f"visual/{args.dataset}/{args.model_name}/image/"
    csv_path = f"visual/{args.dataset}/{args.model_name}/"
    os.makedirs(pred_path, exist_ok=True)

    jsonfile = f'../datasets/{args.dataset}/data_split.json'

    with open(jsonfile, 'r') as f:
        df = json.load(f)
        print(f'testing: {args.dataset}')

    test_files = df['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = BinaryLoader(args.dataset,test_files, get_transform())

    if args.model_name == 'tmunet':
        if args.model_size == 'b':
           dim_group = [32, 64, 128, 256, 512]
        if args.model_size == 's':
           dim_group = [16, 32, 64, 128, 256]
        if args.model_size == 't':
           dim_group = [8, 16, 32, 64, 128]
        model = tmunet.Model(num_classes=1, img_size=args.img_size, embed_dims=dim_group)

    model.load_state_dict(torch.load(args.checkpoints))
    

    model = model.cuda()
    model.eval()
    
    acc_eval = Accuracy()
    pre_eval = Precision()
    dice_eval = Dice()
    recall_eval = Recall()
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    hd_list = []
    image_ids = []
    
    since = time.time()
    
    with torch.no_grad():
        for img, mask, img_id in tqdm(test_dataset):

            img = Variable(torch.unsqueeze(img, dim=0), requires_grad=False).cuda()            
            mask = Variable(torch.unsqueeze(mask, dim=0), requires_grad=False).cuda()

            torch.cuda.synchronize()
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)
            pred = torch.sigmoid(pred)

            pred[pred > 0.5] = 1
            pred[pred <= 0.5] = 0
            
            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()
                     
            if args.debug:
                img_id = list(img_id.split('.'))[0]
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]
                img_numpy[img_numpy==1] = 255 
                cv2.imwrite(f'{pred_path}{img_id}.png',img_numpy)
                
                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy==1] = 255
                cv2.imwrite(f'{pred_path}{img_id}_gt.png',mask_numpy)

            iouscore = iou_eval(pred,mask)
            dicescore = dice_eval(pred,mask)
            hdscore = hd_score(pred,mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
     
            accscore = acc_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu())
            iou_score.append(iouscore.cpu().detach().numpy())
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            image_ids.append(img_id)
            if hdscore != float("inf") and np.isnan(hdscore) != True:
                hd_list.append(hdscore)
            torch.cuda.empty_cache()
            
    time_elapsed = time.time() - since

    result_dict = {'image_id':image_ids, 'miou':iou_score, 'dice':dice_score}
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f'{csv_path}results.csv',index=False)

    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean Dice:',round(np.mean(dice_score),4),round(np.std(dice_score),4))
    print('mean IoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))
    print('mean F1:',round((2*np.mean(pre_score)*np.mean(recall_score))/(np.mean(pre_score)+np.mean(recall_score)),4))
    print('mean accuracy:',round(np.mean(acc_score),4),round(np.std(acc_score),4))
    print('mean precsion:',round(np.mean(pre_score),4),round(np.std(pre_score),4))
    print('mean recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('mean HD:',round(np.mean(hd_list),4),round(np.std(hd_list),4))

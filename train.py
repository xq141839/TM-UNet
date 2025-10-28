import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import copy
import argparse
import os
from dataloader import BinaryLoader
import tmunet
from loss import *
from tqdm import tqdm
import json
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

torch.set_num_threads(4)


def train_model(model, criterion_mask, optimizer, scheduler, num_epochs=5):
    since = time.time()
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()

    best_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
    
            else:
                model.train(False)  

            running_loss_mask = []
            running_corrects_mask = []
            running_loss_boundary = []
            running_corrects_boundary = []
        
            # Iterate over data
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for img, labels, img_id in tqdm(dataloaders[phase]):      
                # wrap them in Variable
    
                img = Variable(img.cuda())
                labels = Variable(labels.cuda())
                                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                pred_mask = model(img)
                pred_mask = torch.sigmoid(pred_mask)

                loss_mask = criterion_mask(pred_mask, labels)
                score_mask = accuracy_metric(pred_mask, labels)

                loss = loss_mask
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss_mask.append(loss_mask.item())
                running_corrects_mask.append(score_mask.item())
             

            epoch_loss = np.mean(running_loss_mask)
            epoch_acc = np.mean(running_corrects_mask)
            
            print('{} Mask Loss: {:.4f} Mask IoU: {:.4f} Totall Loss: {:.4f}'.format(
                phase, np.mean(running_loss_mask), np.mean(running_corrects_mask), epoch_loss))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)


            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f'outputs/{args.model_name}_{args.dataset}_{epoch}.pth')
            if phase == 'valid':
                scheduler.step()
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    return Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='MoNuSeg', help='CVCCDB, MoNuSeg, Ultrasound, ColonDB')
    parser.add_argument('--model_name', type=str,default='tmunet', help='')
    parser.add_argument('--model_size', type=str,default='t', help='')
    parser.add_argument('--img_size', type=int,default=512, help='')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--pretrain', type=str,default='', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'outputs/',exist_ok=True)
    jsonfile = f'../datasets/{args.dataset}/data_split.json'

    with open(jsonfile, 'r') as f:
        df = json.load(f)
        print(f'training: {args.dataset}')
    
    val_files = df['valid']
    train_files = df['train']

    train_dataset = BinaryLoader(args.dataset, train_files, A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.HorizontalFlip(p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ]))
    val_dataset = BinaryLoader(args.dataset, val_files, A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ]))

   
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1)
    dataloaders = {'train':train_loader,'valid':val_loader}

    if args.model_name == 'tmunet':
        if args.model_size == 'b':
           dim_group = [32, 64, 128, 256, 512]
        if args.model_size == 's':
           dim_group = [16, 32, 64, 128, 256]
        if args.model_size == 't':
           dim_group = [8, 16, 32, 64, 128]
        model = tmunet.Model(num_classes=1, img_size=args.img_size, embed_dims=dim_group)
        

    model = model.cuda()

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )

    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')

    total_params = sum(
	param.numel() for param in model.parameters()
    )

    print('Total Params = ' + str(total_params/1000**2) + 'M')

        
    # Loss, IoU and Optimizer
    mask_loss = BinaryMaskLoss()
    accuracy_metric = BinaryIoU()

    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    Loss_list, Accuracy_list = train_model(model, mask_loss, optimizer, exp_lr_scheduler,
                           num_epochs=args.epoch)
    
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')
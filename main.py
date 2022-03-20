# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 10:51:41 2022

@author: NARENDRA
"""

from __future__ import print_function
import argparse
#import sys
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from voxnet import VoxNet
from MCB_B import MCB_B
CLASSES = {0: 'bearing', 1: 'bushing', 2: 'castors_and_wheels', 3: 'clamp', 4: 'disc', 
           5: 'fitting', 6: 'flange', 7: 'fork_joint', 8: 'gear', 9: 'handles', 
           10: 'hinge', 11: 'hook', 12: 'motor', 13: 'nut', 14: 'pin', 15: 'plate',
           16: 'pulley', 17: 'ring', 18: 'rivet', 19: 'rotor', 20: 'screws_and_bolts',
           21: 'spring', 22: 'stud', 23: 'switch', 24: 'washer'}
N_CLASSES = len(CLASSES)

def blue(x): return '\033[94m' + x + '\033[0m'

# Parameter Analysis
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='/home/yelisett/MCB_B', help="dataset path")
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--n-epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
opt = parser.parse_args()
#print(opt)

# Create a directory
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Fixed random seeds
opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Loading Data
train_dataset = MCB_B(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
test_dataset = MCB_B(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')

train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

voxnet = VoxNet(n_classes=N_CLASSES)

print(voxnet)

#Loading weights
if opt.model != '':
    voxnet.load_state_dict(torch.load(opt.model))

# Optimizer
optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
voxnet

num_batch = len(train_dataset) / opt.batchSize
print(num_batch)


for epoch in range(opt.n_epoch):
    for i, sample in enumerate(train_dataloader, 0):
        voxel, cls_idx = sample['voxel'], sample['cls_idx']
        voxel, cls_idx = voxel, cls_idx
        voxel = voxel.float()  # # Voxel is originally int type (0,1), need to be converted to float, torch.Size([256, 1, 64, 64, 64])
   
        # Gradient clearing
        optimizer.zero_grad()
   
        # Network switching training model
        voxnet = voxnet.train()
        pred = voxnet(voxel)  # torch.Size([256, 10])
   
        # Calculating the loss function
   
        loss = F.cross_entropy(pred, cls_idx)
   
        # Back propagation, update weights
        loss.backward()
        optimizer.step()
   
        # Calculate the prediction accuracy of this batch
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(cls_idx.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %
              (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        
        
        # Test every 5 batches
        if i % 5 == 0:
            j, sample = next(enumerate(test_dataloader, 0))
            voxel, cls_idx = sample['voxel'], sample['cls_idx']
            voxel, cls_idx = voxel, cls_idx
            voxel = voxel.float()  # 转loat, torch.Size([256, 1, 64, 64, 64])
            voxnet = voxnet.eval()
            pred = voxnet(voxel)
            loss = F.nll_loss(pred, cls_idx)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(cls_idx.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
                                                             blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
    
    torch.save(voxnet.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


total_correct = 0
total_testset = 0

for i, data in tqdm(enumerate(test_dataloader, 0)):
    voxel, cls_idx = data['voxel'], data['cls_idx']
    voxel, cls_idx = voxel, cls_idx
    voxel = voxel.float()  # float, torch.Size([256, 1, 64, 64, 64])

    voxnet = voxnet.eval()
    pred = voxnet(voxel)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(cls_idx.data).cpu().sum()
    total_correct += correct.item()
    total_testset += voxel.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
    

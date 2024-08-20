# This code was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# it contains the implementation of the loss functions used to train the model
# please Modify the args on the args_****.txt files to control the coef. of each loss value
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

import torch
import warnings
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional, Tuple, Union


class SILogLoss(nn.Module): 
    def __init__(self, args):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.args = args
    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)
               
        if mask is not None:
            input  = input*mask  + 1e-20
            target = target*mask + 1e-20
        g = torch.log(input) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)

class Hist1D_loss(nn.Module):
    def __init__(self, args, t=0.002):
        super(Hist1D_loss, self).__init__()
        self.min = args.min_depth
        self.max = args.max_depth
        self.t = args.t
        self.bins = args.bins_loss if args.bins_loss else (self.max - int(self.min)) + 1
        self.shape = (args.image_height, args.image_width)
        self.int_len = (self.max - self.min) / self.bins
        self.centers = torch.linspace(self.min + self.int_len / 2.0, self.max - self.int_len / 2.0, self.bins).view(1, -1, 1, 1)
    
    def forward(self, gt, pred, mask = None, interpolate = False):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None : 
            pred = pred*mask
            gt   = gt*mask

        h_gt = self.hist(gt)
        h_pred = self.hist(pred)
        emd = torch.mean(torch.abs(h_gt - h_pred) ** 2, axis=[-1]) ** 0.5
        return 10 * torch.mean(emd)

    def hist(self, img):
        d = img - self.centers.to(img.device)
        s = torch.exp(-torch.square(d) / self.t)
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-10)
        h = torch.sum(s, axis=[2, -1])
        h = h / (self.shape[0] * self.shape[1])
        return torch.cumsum(h, -1)
    

class Hist2D_loss(nn.Module):
    def __init__(self, args, t=0.0025):
        super(Hist2D_loss, self).__init__()
        self.name = '2DLoss'

        self.min = args.min_depth
        self.max = args.max_depth
        self.t = args.t
        self.bins = args.bins_loss if args.bins_loss else (self.max - int(self.min)) + 1
        self.shape = (args.image_height, args.image_width)
        self.int_len = (self.max - self.min) / self.bins
        self.centers = torch.linspace(self.min + self.int_len / 2.0, self.max - self.int_len / 2.0, self.bins).view(1, -1, 1, 1)
   
    def forward(self, gt, pred, mask = None, interpolate = False):
        if interpolate:
            pred = nn.functional.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        if mask is not None : 
            #convert inputs
            pred = pred*mask
            gt   = gt*mask
   
        h_gt = self.hist(gt)
        h_pred = self.hist(pred)
        return self.loss(h_pred, h_gt)

    def hist(self, img):
        d = img - self.centers.to(img.device)
        s = torch.exp(-torch.square(d) / self.t)
        s = s / (torch.sum(s, axis=1, keepdims=True) + 1e-6)
        s = s.view(s.shape[0], s.shape[1], -1)
        return s

    def loss(self, h_pred, h_gt):
        p = torch.matmul(h_pred, h_gt.transpose(2, 1)) / (self.shape[0] * self.shape[1])
        py = p.sum(axis=1)
        hy = self.xlogy(py, py).sum(1)
        hxy = self.xlogy(p, p).sum(1).sum(1)
        ce = hy - hxy
        return ce.mean()

    def xlogy(self, x, y):
        log_y = torch.log(torch.clamp(y, min=1e-6))
        res = torch.where(x == 0, torch.zeros_like(x), x * log_y)
        return res

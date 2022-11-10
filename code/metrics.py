import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def dice_score(input:np.array,target:np.array):
    # input:(N,1,x,y,z) numpy
    # target:(N,1,x,y,z) numpy
    # binary
    input[input >= 0.5] = 1
    input[input < 0.5] = 0
    
    N,C = input.shape[0], input.shape[1]
    inputs = input.reshape((N, C, -1))
    targets = target.reshape((N, C, -1))
    intersection = inputs*targets
    smooth = 1e-4
    dice = (smooth+2*intersection.sum(2))/(smooth+ inputs.sum(2)+ targets.sum(2))
    return dice.sum()/(N*C)



def precision_and_sensitive(pred:np.array,label:np.array):
    '''
        pred: (N, class, x, y, z) numpy  0~1
        label: (N, class, x, y, z) numpy  0,1
    '''
    N, C = pred.shape[0], pred.shape[1]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.reshape((N, C, -1))
    label = label.reshape((N, C, -1))

    tp = pred * label
    fp = pred * (1 - label)
    fn = (1 - pred) * label

    pr = (tp.sum(2) + smooth) / (tp.sum(2) + fp.sum(2) + smooth)
    pr = pr.sum() / (N*C)
    rc = (tp.sum(2) + smooth) / (tp.sum(2) + fn.sum(2) + smooth)
    rc = rc.sum() / (N*C)


    return pr, rc



def dice_score_onehot(pred:np.array, mask:np.array):
    '''
        pred: (N, x, y, z)   values:(0,1,2,3)
        mask: (N, 4, x, y, z)
        return: np.array([]*4)
    '''
    C = mask.shape[1]
    dice = [0]*C

    pred = np.expand_dims(pred, axis=1)  # (N, 1, x, y, z)
    mask = np.expand_dims(mask, axis=2)  # (N, 4, 1, x, y, z)
    for i in range(C):
        dice[i] = dice_score(np.where(pred==i, 1, 0), mask[:, i, :, :, :, :])
    
    return np.array(dice)



def precision_and_sensitive_onehot(pred:np.array, mask:np.array):
    '''
        pred: (N, x, y, z)   values:(0,1,2,3)
        mask: (N, 4, x, y, z)
        return: np.array([]*4), np.array([]*4)
    '''
    C = mask.shape[1]
    pr, rc = [0]*C, [0]*C

    pred = np.expand_dims(pred, axis=1)  # (N, 1, x, y, z)
    mask = np.expand_dims(mask, axis=2)  # (N, 4, 1, x, y, z)

    for i in range(C):
        pr[i], rc[i] = precision_and_sensitive(np.where(pred==i, 1, 0), mask[:, i, :, :, :, :])
    
    return np.array(pr), np.array(rc)

    

def multi_class_dice_score(pred:np.array,label:np.array):
    '''
        pred:(N, 3, x, y, z) 0-1
        label:(N, 3, x, y, z)
    '''
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    
    N,C = pred.shape[0], pred.shape[1]
    inputs = pred.reshape((N, C, -1))
    targets = label.reshape((N, C, -1))
    intersection = inputs*targets
    smooth = 1e-4
    dice = (smooth+2*intersection.sum(2))/(smooth+ inputs.sum(2)+ targets.sum(2))       # (N, C)

    return np.mean(dice, axis=0)        # (C, )


def multi_class_precision_and_sensitive(pred:np.array,label:np.array):
    '''
        pred:(N, 3, x, y, z) 0-1
        label:(N, 3, x, y, z) 0,1
    '''
    N, C = pred.shape[0], pred.shape[1]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = pred.reshape((N, C, -1))
    label = label.reshape((N, C, -1))

    tp = pred * label
    fp = pred * (1 - label)
    fn = (1 - pred) * label

    pr = (tp.sum(2) + smooth) / (tp.sum(2) + fp.sum(2) + smooth)
    pr = np.mean(pr, axis=0)        # (C, )
    rc = (tp.sum(2) + smooth) / (tp.sum(2) + fn.sum(2) + smooth)
    rc = np.mean(rc, axis=0)        # (C, )

    return pr, rc



class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input:torch.Tensor, target:torch.Tensor):
        # input: (N,c,x,y,z) torch 
        # target:(N,c,x,y,z) torch 0~1
        input = torch.sigmoid(input)
        N, C = input.shape[0], input.shape[1]
        input, target = input.view(N,C,-1), target.view(N,C,-1)
        smooth = 1e-4
        intersection = input*target
        dice = 2*(torch.sum(intersection) + smooth) / (torch.sum(input) + torch.sum(target) + smooth)       # (N,C)
        return 1 - torch.mean(dice)



class Dice_BCEWithLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,input:torch.Tensor, target:torch.Tensor):
        # input: (N,c,x,y,z) torch 
        # target:(N,c,x,y,z) torch 0~1

        bceloss = nn.BCEWithLogitsLoss()(input,target)
        diceloss = DiceLoss()(input,target) 
        return  bceloss + diceloss 



class MultiHead_Dice_BCEWithLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._loss = Dice_BCEWithLogits()
    
    def forward(self, input:torch.Tensor, target:torch.Tensor):
        # input: (N, c, x, y, z), (N, c, x/2, y/2, z/2), (N, c, x/4, y/4, z/4)
        # target:(N, c, x, y, z)
        mask1 = F.interpolate(target, scale_factor=0.5, mode='nearest')
        mask2 = F.interpolate(mask1, scale_factor=0.5, mode='nearest')

        loss = self._loss(input[0], target) + 0.5*self._loss(input[1], mask1) + 0.25*self._loss(input[2], mask2)
        return loss




class Dice_VAE(nn.Module):
    def __init__(self, txt_path) -> None:
        super().__init__()
        self.path = txt_path

    def forward(self,epoch:int, pred:torch.Tensor, mask:torch.Tensor, vae:torch.Tensor, image:torch.Tensor, mu:torch.Tensor, logsigma:torch.Tensor,\
         if_print):
        # input: (N,4,x,y,z) torch 
        # pred:(N,3,x,y,z) torch
        
        bcefunc = nn.BCEWithLogitsLoss()
        bceloss = bcefunc(pred,mask)
        pred = nn.Sigmoid()(pred)
        diceloss = DiceLoss(pred,mask) 

        l2_weight = 1
        l2loss = F.mse_loss(vae, image, reduction='mean') # 一般在 40
        kl_weight = 0.00025
        kl_div = torch.mean(-0.5 * torch.sum(1 + logsigma - mu ** 2 - logsigma.exp(), dim=1), dim=0)    # 一般在900

        # 加权
        # a, b, c = 1*(epoch//10+1), 10/(epoch//10+1), 10/(epoch//10+1)
        a, b, c = 0, 1, 4
        a, b, c = a/(a+b+c), b/(a+b+c), c/(a+b+c)

        if if_print:
            with open(self.path,'a') as f:
                f.write(f'\n\n\n\nBCE:{bceloss}\tDice:{diceloss}\tMSE_Loss:{l2_weight*l2loss}\tKL_Loss:{kl_weight*kl_div}')
                f.write(f'\nWeight==>\tBCE:{a/2}\tDice:{a/2}\tMSE_Loss:{b}\tKL_Loss:{c}')
        return a/2*bceloss , a/2*diceloss , b*l2_weight*l2loss , c*kl_weight*kl_div




class VAE_classify(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,origin:torch.Tensor, preds:Tuple, is_mask):
        '''
            preds: (pred(0~1), vae, mu, logsigma)
        '''
        loss1 = (torch.mean(preds[0]) + 1) /2  if is_mask else (-torch.mean(preds[0]) + 1) /2
        l2_weight = 2.5e-2
        l2loss = F.mse_loss(preds[1], origin, reduction='mean') 
        kl_weight = 1e-3
        kl_div = torch.mean(-0.5 * torch.sum(1 + preds[3] - preds[2] ** 2 - preds[3].exp(), dim=1), dim=0) 
                
        return loss1 + 0.1*l2_weight*l2loss + 0.1*kl_weight*kl_div


    
class cGANLoss(nn.Module):
    def __init__(self)->None:
        super().__init__()

    def forward(self, inputs:torch.Tensor, True_or_Fake:bool):
        '''
            inputs: the output of discriminator (N, c, x, h, d)
        '''
        if False:
            valid, invalid = 0.1, 0.9
            target_tensor = torch.tensor(valid, dtype=torch.float32).expand_as(inputs).to(inputs.device) if True_or_Fake==True else torch.tensor(invalid, dtype=torch.float32).expand_as(inputs).to(inputs.device)
            cganloss = nn.BCEWithLogitsLoss()(inputs, target_tensor)
            return cganloss

        # contrastive   loss
        if True:
            loss = -torch.mean(inputs) 
            return loss if True_or_Fake==True else -loss

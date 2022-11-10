import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np









def show_img_to_wandb(input,is_mask=True):
    # input: 256*256 numpy
    if is_mask:
        # 0~1  --> binary --> 0~255
        input_cut = np.copy(input)
        input_cut[np.nonzero(input_cut<0.5)] = 0.0
        input_cut[np.nonzero(input_cut>=0.5)] = 255.0
        return input_cut
    else:
        # mean std image --> 0~255
        input_min,input_max = input.min(), input.max()
        input = (input - input_min) / (input_max - input_min) * 255.0
        return input


def show_9img_to_wandb(input):
    # input: (9,256,256) numpy
    nclass = len(input)
    imgs = np.zeros_like(input[0])  #(256,256)
    for i in range(nclass-1):   # do not compute background
        # 0~1  --> binary --> 0~255
        img = input[i,:,:]
        img_cut = np.copy(img)
        img_cut[np.nonzero(img_cut<0.5)] = 0.0
        img_cut[np.nonzero(img_cut>=0.5)] = 255.0
        imgs = imgs + img_cut
    imgs = np.clip(imgs,0.0 ,255.0)
    return imgs



def make_imgs_to_wandb(a,b,c,d):
    # a,b,c,d:(256,256) numpy 0~255
    plant = np.ones((256,256*4+50*3))*255
    pos = 0
    plant[:,pos:(pos+256)] = a
    pos += 256+50
    plant[:,pos:(pos+256)] = b
    pos += 256+50
    plant[:,pos:(pos+256)] = c
    pos += 256+50
    plant[:,pos:(pos+256)] = d
    return plant





def masking(origin:torch.Tensor, pred:torch.Tensor):
    '''
        origin: (N, 4, x, y, z)
        pred:   (N, 1, x, y, z)  0~1
        return: (N, 4, x, y, z)
    '''
    device = origin.device
    output_masked = origin.clone().float().to(device)
    pred_clone = pred[:,-1,:,:,:].clone().float()
    for modality in range(origin.shape[1]):
        output_masked[:,modality,:,:,:] = origin[:,modality,:,:,:] * pred_clone
    return output_masked




def multi_class_masking(origin:torch.Tensor, pred:torch.Tensor):
    '''
        origin: (N, 4, x, y, z)
        pred:   (N, 3, x, y, z)  0~1
        return: (N, 12, x, y, z)
    '''
    C = pred.shape[1]
    output_masked = None
    for i in range(C):  
        tmp = masking(origin, pred[:,i,:,:,:].unsqueeze(1))     # (N, 4, x, y, z)
        if output_masked==None:
            output_masked = tmp
        else:
            output_masked = torch.concat((output_masked, tmp), dim=1)   # (N, 4*(i+1), x, y, z)
    
    return output_masked

import random
import numpy as np
from torchvision.transforms import transforms
import pickle
from scipy import ndimage
import os
import torch




class MaxMinNormalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        C = image.shape[0]
        for i in range(C):
            Max = np.max(image[i,:,:,:])
            Min = np.min(image[i,:,:,:])
            image[i,:,:,:] = (image[i,:,:,:] - Min) / (Max - Min)

        return {'image': image, 'label': label}


class Normalization(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        C = image.shape[0]
        image_normalized = np.zeros((1, image.shape[1], image.shape[2], image.shape[3])).astype('float')            # 一定要输出float型
        for i in range(C):
            y = image[i,:,:,:]
            mu, std = y.mean(), y.std()
            y = (y - mu) / std
            Max, Min = np.max(y), np.min(y)
            normalized_y = (y - Min) / (Max - Min) *2 + (-1)      # value (-1, 1)
            image_normalized = np.append(image_normalized, normalized_y[None], axis=0)


        return {'image': image_normalized[1:,:,:,:], 'label': label}


class Pad(object):
    def __call__(self, sample):
        '''
            image, label: (c, 240, 240, 155)
        '''
        image = sample['image']
        label = sample['label']

        image = np.pad(image, ((0, 0), (0, 0), (0, 0), (0, 5)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}
    #(240,240,155)>(240,240,160)




class Random_Flip(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = np.flip(image, 3)
            label = np.flip(label, 3)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)

        return {'image': image, 'label': label}


class Random_Crop(object):
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        assert image.shape[3] >= self.size[2]

        delta_x = image.shape[1] - self.size[0]
        delta_y = image.shape[2] - self.size[1]
        delta_z = image.shape[3] - self.size[2]

        H = random.randint(0, delta_x)
        W = random.randint(0, delta_y)
        D = random.randint(0, delta_z)
        # H = random.randint(0, 2) * (delta_x//2)
        # W = random.randint(0, 2) * (delta_y//2)
        # D = random.randint(0, 2) * (delta_z//2)
        

        image = image[:, H: H + self.size[0], W: W + self.size[1], D: D + self.size[2]]
        label = label[:, H: H + self.size[0], W: W + self.size[1], D: D + self.size[2]]

        return {'image': image, 'label': label}




class Center_Crop(object):
    def __init__(self, size) -> None:
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        
        assert image.shape[3] >= self.size[2]

        delta_x = image.shape[1] - self.size[0]
        delta_y = image.shape[2] - self.size[1]
        delta_z = image.shape[3] - self.size[2]


        H = delta_x//2
        W = delta_y//2
        D = delta_z//2
        

        image = image[:, H: H + self.size[0], W: W + self.size[1], D: D + self.size[2]]
        label = label[:, H: H + self.size[0], W: W + self.size[1], D: D + self.size[2]]

        return {'image': image, 'label': label}



class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        image = sample['image']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=image.shape)
        shift_factor = np.random.uniform(-factor, factor, size=image.shape)

        image = image*scale_factor+shift_factor

        return {'image': image, 'label': label}




class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image)
        label = sample['label']
        label = np.ascontiguousarray(label)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        return {'image': image, 'label': label}


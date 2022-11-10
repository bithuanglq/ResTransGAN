from torch.utils.data import Dataset
import numpy as np
import os
import json
import nibabel as nib
from data_augmentation import *



def split_dataset_2015(data_path,per=0.7):
    ''' 
        随机划分训练集、验证集    BraTS 2015
    '''
    with open(data_path) as json_file:
        data = json.load(json_file)
        length = len(data.keys())
        if per == 1:
            print("Using the whole dataset for training!\n")
            train_set = np.arange(length)
            np.random.shuffle(train_set)
            val_set = train_set.copy()          # train_set == val_set == the whole dataset

        else:
            arr_HGG, arr_LGG = np.arange(220), np.arange(length-220)
            np.random.shuffle(arr_HGG)
            train_set_HGG, val_set_HGG = arr_HGG[:int(len(arr_HGG)*per)] + length - 220, arr_HGG[int(len(arr_HGG)*per):] + length - 220
            np.random.shuffle(arr_LGG)
            train_set_LGG, val_set_LGG = arr_LGG[:int(len(arr_LGG)*per)], arr_LGG[int(len(arr_LGG)*per):]
            train_set, val_set = np.append(train_set_LGG, train_set_HGG), np.append(val_set_LGG, val_set_HGG)
            np.random.shuffle(train_set)
            np.random.shuffle(val_set)
    return train_set, val_set

  
  
class MyDataset2_2015(Dataset):
    '''
        load data from original .npy files  BraTS2015
    '''
    def __init__(self, datapath:str, jsonpath:str, index_set:np.array) -> None:
        super().__init__()
        with open(jsonpath) as json_file:
            self.json  = json.load(json_file)
        self.set = index_set
        self.datapath = datapath
        self.transforms = transforms.Compose([
            Pad(),
            Random_Crop(size=(160, 192, 160)),
            Random_Flip(),
            Normalization(),
            ToTensor()
        ])

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, index):
        index = self.set[index]
        data = np.load(os.path.join(self.datapath,   'VSD.' + self.json[str(index)]['flair'].split('/')[-3] + '.' + self.json[str(index)]['flair'].split('/')[4].split('.')[-1] + '.npy') )    # (4+4+4, 160, 192, 160)
        image, mask = data[:4,:,:,:], data[4:,:,:,:]            # WT, TC, ET
        sample = {'image':image, 'label':mask}
        sample = self.transforms(sample)
        image, mask = sample['image'], sample['label']

        return image, mask, np.array([index])

      
     
class MyValidDataset2_2015(Dataset):
    '''
        load data from original .npy files  BraTS2015
    '''
    def __init__(self, datapath:str, jsonpath:str, index_set:np.array) -> None:
        super().__init__()
        with open(jsonpath) as json_file:
            self.json  = json.load(json_file)
        self.set = index_set
        self.datapath = datapath
        self.transforms = transforms.Compose([
            Pad(),
            Center_Crop(size=(160, 192, 160)),          # 不同之处
            Random_Flip(),
            Normalization(),
            ToTensor()
        ])

    def __len__(self):
        return len(self.set)
    
    def __getitem__(self, index):
        index = self.set[index]
        data = np.load(os.path.join(self.datapath,   'VSD.' + self.json[str(index)]['flair'].split('/')[-3] + '.' + self.json[str(index)]['flair'].split('/')[4].split('.')[-1] + '.npy') )    # (4+4+4, 160, 192, 160)
        image, mask = data[:4,:,:,:], data[4:,:,:,:]            # WT, TC, ET
        sample = {'image':image, 'label':mask}
        sample = self.transforms(sample)
        image, mask = sample['image'], sample['label']

        return image, mask, np.array([index])


  
  

class MyTestDataset2_2015(Dataset):
    '''
        load data from original .npy files  BraTS2015
    '''
    def __init__(self, datapath:str, jsonpath:str) -> None:
        super().__init__()
        with open(jsonpath) as json_file:
            self.json  = json.load(json_file)
        self.datapath = datapath
        self.transforms = transforms.Compose([
            Pad(),
            ToTensor()
        ])

    def __len__(self):
        return len(self.json.keys())
    
    def __getitem__(self, index):
        # image (4, 240, 240, 160)
        data = np.load(os.path.join(self.datapath,   'VSD.' + self.json[str(index)]['flair'].split('/')[-3] + '.' + self.json[str(index)]['flair'].split('/')[4].split('.')[-1] + '.npy') )    
        image = data[:4, :,:,:]
        sample = {'image':image, 'label':image}
        sample = self.transforms(sample)
        image = sample['image']

        return image, np.array([index])

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import random
from tqdm import tqdm
import SimpleITK as sitk
from typing import List
import json


from utils import *
from dataloader import *
from TransBTS.TransBTS_downsample8x_skipconnection import My_TransBTS

# Hyper parameters
BATCH_SIZE = 4
main_gpu = 0
DATAPARALLEL = True
PTHS = 1



work_path = '/home/xx/Project/BraTS'
data_path = './BraTS2015/Testing/data.json'
case_path = 'checkpoint_2015/test_case/ResTransGAN_with_cnn_resnet/final'
weight_path = 'checkpoint_2015/model/ResTransGAN_with_cnn_resnet/epoch_42_rank0.pth'    # 117
os.chdir(work_path)
print(os.getcwd())



# GPU CUDA Parallel
if DATAPARALLEL:
    # GPUs = [4,5,6,7]
    GPUs = [0,1,2,3]
    main_gpu = GPUs[0]
torch.cuda.set_device('cuda:{}'.format(main_gpu))
device = torch.device('cuda:{}'.format(main_gpu) if torch.cuda.is_available() else 'cpu')
print("device=",device)
torch.backends.cudnn.benchmark = True   # 加快训练



def set_random_seed(seed=10):
    random.seed(seed)                  # random 模块中的随机种子，random是python中用于产生随机数的模块
    np.random.seed(seed)               # numpy中的随机种子
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True   # deterministic用来固定内部随机性，即每次开机训练输入相同则输出大致相同



def save_test_case(out:np.array, save_folder:str, index:np.array):
    '''
        save pred  .mha to save_folder/
        out: (N, 3, X=240, Y=240, Z=155)
        index:(N, )
    '''
    with open(data_path) as json_file:
        json_data  = json.load(json_file)

    
    def save_nii(seg, seg_nii_path, config):
        '''
            seg: (4, 240, 240, 155)
        '''
        seg[seg >= 0.5] = 1
        seg[seg < 0.5] = 0
        WT = seg[0]
        TC = seg[1]
        ET = seg[2]
        NE = np.zeros_like(ET)
        NE[(ET>0.3) & (ET<0.5)] = 1

        final = np.zeros_like(ET)
        final[(WT==1) & (TC==1) & (ET==1) & (NE==1)] = 1
        final[(WT==1) & (TC==0)] = 2
        final[(WT==1) & (TC==1) & (ET==0)] = 3
        final[(WT==1) & (TC==1) & (ET==1) & (NE==0)] = 4
        final = final.astype('short')    

        origin_seg = final
        origin_seg = sitk.GetImageFromArray(np.transpose(origin_seg,(2,1,0)))
        origin_seg.SetSpacing(config['spacing'])
        origin_seg.SetOrigin(config['origin'])
        origin_seg.SetDirection(config['direction'])
        sitk.WriteImage(origin_seg, os.path.join(save_folder, seg_nii_path))
    
    for i in range(out.shape[0]):
        idx = str(index[i][0])
        name = 'VSD.' + json_data[idx]['flair'].split('/')[-3] + '.' + json_data[idx]['flair'].split('/')[-2].split('.')[-1]
        config = {
                "spacing": json_data[idx]['spacing'],
                "origin": json_data[idx]['origin'],
                "direction": json_data[idx]['direction']
        }
        save_nii(out[i], name +'.mha', config)



def ensemble_inference_v1(net, image:torch.Tensor, size:Tuple[int,int,int]):
    '''
        image: (N, 4, 240, 240, 160)
        size:  input size of net  (160, 192, 160)
        return: (N, 3, 240, 240, 155)
    '''
    device = image.device
    origin_shape, new_shape = (image.shape[2], image.shape[3], image.shape[4]), size
    delta_x, delta_y = origin_shape[0]-new_shape[0], origin_shape[1]-new_shape[1]
    
    assert delta_x > 0
    assert delta_y > 0

    numbers = torch.zeros((image.shape[0], 3, 240, 240, 160)).to(device)
    output =  torch.zeros((image.shape[0], 3, 240, 240, 160)).to(device)
    # start_coordinates = [(0, 0, 0),                     # 四个顶点 + 一个中心
    #                      (0, delta_y, 0), 
    #                      (delta_x, 0, 0),  
    #                      (delta_x, delta_y, 0), 
    #                      (delta_x//2, delta_y//2, 0)]
    start_coordinates = [(delta_x//2, delta_y//2, 0)]


    for x,y,z in start_coordinates:
        xx, yy, zz = x+new_shape[0], y+new_shape[1], z+new_shape[2]
        numbers[:,:, x:xx, y:yy, z:zz] += 1
        inputs = image[:,:, x:xx, y:yy, z:zz].float().clone()           # clone() 否则循环会改变image值
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                input = inputs[i,j,:,:,:]
                input = (input - torch.mean(input)) / torch.std(input)
                Max, Min = torch.max(input), torch.min(input)
                inputs[i,j,:,:,:] = (input - Min) / (Max - Min) * 2 + (-1)              # values:(-1,1)
        tmp_output = net(inputs)[0]         # [0]          
        tmp_output = torch.sigmoid(tmp_output)
        tmp_output[tmp_output>=0.5] = 1
        tmp_output[tmp_output<0.5] = 0  
        output[:,:, x:xx, y:yy, z:zz] += tmp_output         
    

    numbers[numbers==0] = 1             # 防止报错
    assert torch.min(numbers) > 0           # torch.unique(numbers) ==> tensor([1, 2, 3, 4, 5, 9])

    output /= numbers
    output[output>=0.5] = 1
    output[output<0.5] = 0

    return output[:, :, :, :, :155]







if __name__=='__main__':
    set_random_seed(10) # 设置随机数种子
    test_dataloader = DataLoader(MyTestDataset2_2015('preprocessed_data_2015/validation', data_path),batch_size=BATCH_SIZE,shuffle=True, num_workers=BATCH_SIZE)
    print('Load Data Successfully!')

    net =  My_TransBTS(in_channels=4, n_classes=3)


    if DATAPARALLEL:
        net = nn.DataParallel(net.to(device), device_ids=GPUs,output_device=main_gpu)
    else:
        net = net.to(device)




    with tqdm(test_dataloader,desc=f'Inference_v2',unit='img',unit_scale=True) as pbar2:
        pbar2.write(f'----开始验证：-----')
        net.eval()
        with torch.no_grad():
            for i, (image, index) in enumerate(pbar2):
                image = image.float().cuda(non_blocking=True)
                out = torch.zeros((image.shape[0], 3, 240, 240, 155)).float().cuda(non_blocking=True)
                for idx in range(PTHS):
                    tmp_weight_path = weight_path[:-5] + str(idx) + '.pth'
                    checkpoint = torch.load(tmp_weight_path, map_location=device)
                    # checkpoint = torch.load(weight_path, map_location=device)
                    net.load_state_dict(checkpoint['generator'])
                    tmp_out = ensemble_inference_v1(net, image, size=(160, 192, 160))
                    out = out + tmp_out
                out = out / PTHS        # (4, 3, 240, 240, 155)


                # save test case
                if not os.path.exists(case_path):
                    os.system(f'mkdir {case_path}')
                save_test_case(out.cpu().numpy(), case_path, index=index.cpu().numpy())

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DataSampler
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
import random
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from collections import Counter
import GPUtil
import warnings
import SimpleITK as sitk


from dataloader import *
from metirc import *
from discriminator import *
from utils import *
from TransBTS.TransBTS_downsample8x_skipconnection import My_TransBTS


# DDP 版本   
# 指令： nohup python -m torch.distributed.launch --nproc_per_node=4 your_python_file_path > nohup.out 2>&1 &

# Hyper parameters
BATCH_SIZE = 1
LR_INITIAL_D, LR_INITIAL_G = 1e-4, 1e-4
EPOCH = 210
INTERVAL = 3
CLIP_VALUE = 0.05
main_gpu = 0
DATAPARALLEL = True
SAVE_PARAMS = True



work_path = '/home/xx/Project/BraTS'
data_path = './BraTS2015/BRATS2015_Training/data.json'
save_path = './checkpoint_2015/model/1G_3D'           # save model params
txt_path = './checkpoint_2015/1G_3D.txt'           
weight_path = None
os.chdir(work_path)
print(os.getcwd())



# GPU CUDA Parallel
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend='nccl')
device = torch.device('cuda', local_rank)
torch.backends.cudnn.benchmark = True   # 加快训练
print(f"[init] == local rank: {local_rank}, global rank: {rank}")





def set_random_seed(seed=10):
    random.seed(seed)                  # random 模块中的随机种子，random是python中用于产生随机数的模块
    np.random.seed(seed)               # numpy中的随机种子
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True   # deterministic用来固定内部随机性，即每次开机训练输入相同则输出大致相同






if __name__ == '__main__':
    set_random_seed(10) # 设置随机数种子
    train_set, val_set = split_dataset_2015(data_path=data_path, per=0.9)
    train_dataset = MyDataset2_2015('preprocessed_data_2015/training', data_path, train_set)
    val_dataset = MyValidDataset2_2015('preprocessed_data_2015/training', data_path, val_set)
    train_sampler = DataSampler(train_dataset, shuffle=True)
    val_sampler = DataSampler(val_dataset, shuffle=True)
    # 注意 num_workers 个数， pin_memory=False  估计因为锁存不够
    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=BATCH_SIZE, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=BATCH_SIZE, sampler=val_sampler)
    if rank == 0:  print('Load Data Successfully!')
        

    generator =  My_TransBTS(in_channels=4, n_classes=3)
    discriminator1 = CNN(in_channels=4)
    discriminator2 = CNN(in_channels=4)
    discriminator3 = CNN(in_channels=4)


    generator = generator.to(device)
    discriminator1 = discriminator1.to(device)
    discriminator2 = discriminator2.to(device)
    discriminator3 = discriminator3.to(device)
    # 参考：https://zhuanlan.zhihu.com/p/409117481
    generator = DDP(generator, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    discriminator1 = DDP(discriminator1, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    discriminator2 = DDP(discriminator2, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    discriminator3 = DDP(discriminator3, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)

    if weight_path:
        checkpoint = torch.load(weight_path[rank])
        generator.load_state_dict(checkpoint['generator'])
        discriminator1.load_state_dict(checkpoint['discriminator1'])
        discriminator2.load_state_dict(checkpoint['discriminator2'])
        discriminator3.load_state_dict(checkpoint['discriminator3'])
        if rank == 0:  print('Load Model Successfully!')
        del checkpoint
        torch.cuda.empty_cache()

    optimizer_G = optim.RMSprop(generator.parameters(), lr=LR_INITIAL_G)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, 70, gamma=0.1)
    optimizer_D1 = optim.RMSprop(discriminator1.parameters(), lr=LR_INITIAL_D)
    scheduler_D1 = torch.optim.lr_scheduler.StepLR(optimizer_D1, 70, gamma=0.1)
    optimizer_D2 = optim.RMSprop(discriminator2.parameters(), lr=LR_INITIAL_D)
    scheduler_D2 = torch.optim.lr_scheduler.StepLR(optimizer_D2, 70, gamma=0.1)
    optimizer_D3 = optim.RMSprop(discriminator3.parameters(), lr=LR_INITIAL_D)
    scheduler_D3 = torch.optim.lr_scheduler.StepLR(optimizer_D3, 70, gamma=0.1)



    if rank == 0:
        with open(txt_path,'w') as f:
            f.write('Start....\n')

        for epoch in range(EPOCH):
            # train
            if True:
                with tqdm(train_dataloader,desc=f'Training--{epoch}',unit='img',unit_scale=True) as pbar:
                    pbar.write(f'--------------开始第{epoch}轮！-------------')
                    generator.train()
                    Losses_D, Losses_G, len_D, len_G = 0,0,1e-4, 1e-4
                    for i, (image, mask, _) in enumerate(pbar):
                        image,mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)

                        # ---------------------
                        #  Train Discriminator1
                        # ---------------------
                        
                        optimizer_D1.zero_grad()
                        fake_imgs = generator(image)[0].detach()       # detach() 生成器不反向传播
                        fake_imgs = torch.sigmoid(fake_imgs)
                        result1 = discriminator1(multi_class_masking(image, fake_imgs[:, 0:1, :, :, :]))
                        mask_D1 = discriminator1(multi_class_masking(image, mask[:, 0:1, :, :, :]))
                        loss_D1 = -torch.mean(torch.abs(result1 - mask_D1))/3
                        loss_D1.backward()
                        optimizer_D1.step()
                        Losses_D += loss_D1.item()
                        # clip parameters in D
                        for p in discriminator1.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                        # delete
                        del result1, mask_D1, loss_D1
                        torch.cuda.empty_cache()

                        # ---------------------
                        #  Train Discriminator2
                        # ---------------------
                        
                        optimizer_D2.zero_grad()
                        result2 = discriminator2(multi_class_masking(image, fake_imgs[:, 1:2, :, :, :]))
                        mask_D2 = discriminator2(multi_class_masking(image, mask[:, 1:2, :, :, :]))
                        loss_D2 = -torch.mean(torch.abs(result2 - mask_D2))/3
                        loss_D2.backward()
                        optimizer_D2.step()
                        Losses_D += loss_D2.item()
                        # clip parameters in D
                        for p in discriminator2.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                        # delete
                        del result2, mask_D2, loss_D2
                        torch.cuda.empty_cache()

                        # ---------------------
                        #  Train Discriminator3
                        # ---------------------
                        
                        optimizer_D3.zero_grad()
                        result3 = discriminator3(multi_class_masking(image, fake_imgs[:, 2:3, :, :, :]))
                        mask_D3 = discriminator3(multi_class_masking(image, mask[:, 2:3, :, :, :]))
                        loss_D3 = -torch.mean(torch.abs(result3 - mask_D3))/3
                        loss_D3.backward()
                        optimizer_D3.step()
                        Losses_D += loss_D3.item()
                        # clip parameters in D
                        for p in discriminator3.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                        # delete
                        del result3, mask_D3, loss_D3
                        torch.cuda.empty_cache()

                        len_D += 1
                        # -----------------
                        #  Train Generator
                        # -----------------

                        optimizer_G.zero_grad()
                        gen_imgs = generator(image)      # Generate a batch of images
                        dice_loss = MultiHead_Dice_BCEWithLogits()(gen_imgs, mask)
                        gen_imgs = torch.sigmoid(gen_imgs[0])
                        result1 = discriminator1(multi_class_masking(image, fake_imgs[:, 0:1, :, :, :]))
                        mask_G1 = discriminator1(multi_class_masking(image, mask[:, 0:1, :, :, :]))
                        ad_loss = torch.mean(torch.abs(result1 - mask_G1)) / 3
                        # # delete
                        del result1, mask_G1
                        torch.cuda.empty_cache() 
                        result2 = discriminator2(multi_class_masking(image, fake_imgs[:, 1:2, :, :, :]))
                        mask_G2 = discriminator2(multi_class_masking(image, mask[:, 1:2, :, :, :]))
                        ad_loss += torch.mean(torch.abs(result2 - mask_G2)) / 3
                        # # delete
                        del result2, mask_G2
                        torch.cuda.empty_cache() 
                        result3 = discriminator3(multi_class_masking(image, fake_imgs[:, 2:3, :, :, :]))
                        mask_G3 = discriminator3(multi_class_masking(image, mask[:, 2:3, :, :, :]))
                        ad_loss += torch.mean(torch.abs(result3 - mask_G3)) / 3
                        # delete
                        del result3, mask_G3
                        torch.cuda.empty_cache() 
                        loss_G = dice_loss + ad_loss
                        loss_G.backward()
                        optimizer_G.step()

                        Losses_G += loss_G.item()
                        len_G += 1


                    with open(txt_path,'a') as f:
                        f.write(f'\n\nepoch:{epoch}\tloss_D:{Losses_D/len_D}\tloss_G:{Losses_G/len_G}\n')
                        lr_D, lr_G = optimizer_D1.param_groups[0]['lr'], optimizer_G.param_groups[0]['lr']
                        f.write(f'learning rate:\tdiscriminator1,2,3:{lr_D}\tgenerator:{lr_G}\n')
                    # 保存参数
                    if SAVE_PARAMS and (epoch%INTERVAL) == 0:
                        state = {'generator':generator.state_dict(), 'discriminator1':discriminator1.state_dict(), 'discriminator2':discriminator2.state_dict(), 'discriminator3':discriminator3.state_dict(), 
                                 'GPU':[rank]}
                        if not os.path.exists(save_path):
                            os.system(f'mkdir {save_path}')
                        torch.save(state, f'{save_path}/epoch_{epoch}_{rank}.pth')
                        pbar.write(f'--------------模型 epoch_{epoch}.pth 保存完成！-------------')


            scheduler_G.step()
            scheduler_D1.step()
            scheduler_D2.step()
            scheduler_D3.step()


            #  validation
            if (epoch%INTERVAL) == 0:
                with tqdm(val_dataloader,desc=f'Validation--{epoch}',unit='img',unit_scale=True) as pbar2:
                    pbar2.write(f'----开始验证：-----')
                    generator.eval()
                    dice, len_val = np.zeros(3), 0
                    pr, rc = np.zeros(3), np.zeros(3)
                    with torch.no_grad():
                        for i, (val_image, val_mask, index) in enumerate(pbar2):
                            val_image,val_mask = val_image.float().cuda(non_blocking=True), val_mask.float().cuda(non_blocking=True)

                            val_out = generator(val_image)[0]     # (N, 3, x, y, z)
                            val_out = torch.sigmoid(val_out)
                            tmp_dice = multi_class_dice_score(val_out.cpu().numpy(), val_mask.cpu().numpy())
                            tmp_pr, tmp_rc = multi_class_precision_and_sensitive(pred=val_out.cpu().numpy(), label=val_mask.cpu().numpy())
                            dice = dice + tmp_dice
                            pr = pr + tmp_pr
                            rc = rc +  tmp_rc
                            len_val += 1
                        # delete
                        del val_out, val_mask, val_image
                        torch.cuda.empty_cache()

                        dice, pr, rc = dice/len_val, pr/len_val, rc/len_val
                        with open(txt_path,'a') as f:
                            f.write(f'epoch:{epoch}\tlocal_rank:{local_rank}\nClass:[WT, TC, ET]\tMean Dice:{dice.mean()}\nVal Dice Score:{dice}\tPrecision:{pr}\tSensitive:{rc}\n')

    else:
        for epoch in range(EPOCH):
            # train
            generator.train()
            Losses_D, Losses_G, len_D, len_G = 0,0,1e-4, 1e-4
            for i, (image, mask, _) in enumerate(train_dataloader):
                image,mask = image.float().cuda(non_blocking=True), mask.float().cuda(non_blocking=True)

                # ---------------------
                #  Train Discriminator1
                # ---------------------
                
                optimizer_D1.zero_grad()
                fake_imgs = generator(image)[0].detach()       # detach() 生成器不反向传播
                fake_imgs = torch.sigmoid(fake_imgs)
                result1 = discriminator1(multi_class_masking(image, fake_imgs[:, 0:1, :, :, :]))
                mask_D1 = discriminator1(multi_class_masking(image, mask[:, 0:1, :, :, :]))
                loss_D1 = -torch.mean(torch.abs(result1 - mask_D1))/3
                loss_D1.backward()
                optimizer_D1.step()
                Losses_D += loss_D1.item()
                # clip parameters in D
                for p in discriminator1.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                # delete
                del result1, mask_D1, loss_D1
                torch.cuda.empty_cache()

                # ---------------------
                #  Train Discriminator2
                # ---------------------
                
                optimizer_D2.zero_grad()
                result2 = discriminator2(multi_class_masking(image, fake_imgs[:, 1:2, :, :, :]))
                mask_D2 = discriminator2(multi_class_masking(image, mask[:, 1:2, :, :, :]))
                loss_D2 = -torch.mean(torch.abs(result2 - mask_D2))/3
                loss_D2.backward()
                optimizer_D2.step()
                Losses_D += loss_D2.item()
                # clip parameters in D
                for p in discriminator2.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                # delete
                del result2, mask_D2, loss_D2
                torch.cuda.empty_cache()

                # ---------------------
                #  Train Discriminator3
                # ---------------------
                
                optimizer_D3.zero_grad()
                result3 = discriminator3(multi_class_masking(image, fake_imgs[:, 2:3, :, :, :]))
                mask_D3 = discriminator3(multi_class_masking(image, mask[:, 2:3, :, :, :]))
                loss_D3 = -torch.mean(torch.abs(result3 - mask_D3))/3
                loss_D3.backward()
                optimizer_D3.step()
                Losses_D += loss_D3.item()
                # clip parameters in D
                for p in discriminator3.parameters():                # Clip weights of discriminator  裁剪分辨器D的权重是WGAN的特点，加速收敛
                    p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)
                # delete
                del result3, mask_D3, loss_D3
                torch.cuda.empty_cache()

                len_D += 1
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()
                gen_imgs = generator(image)      # Generate a batch of images
                dice_loss = MultiHead_Dice_BCEWithLogits()(gen_imgs, mask)
                gen_imgs = torch.sigmoid(gen_imgs[0])
                result1 = discriminator1(multi_class_masking(image, fake_imgs[:, 0:1, :, :, :]))
                mask_G1 = discriminator1(multi_class_masking(image, mask[:, 0:1, :, :, :]))
                ad_loss = torch.mean(torch.abs(result1 - mask_G1)) / 3
                # delete
                del result1, mask_G1
                torch.cuda.empty_cache() 
                result2 = discriminator2(multi_class_masking(image, fake_imgs[:, 1:2, :, :, :]))
                mask_G2 = discriminator2(multi_class_masking(image, mask[:, 1:2, :, :, :]))
                ad_loss += torch.mean(torch.abs(result2 - mask_G2)) / 3
                # delete
                del result2, mask_G2
                torch.cuda.empty_cache() 
                result3 = discriminator3(multi_class_masking(image, fake_imgs[:, 2:3, :, :, :]))
                mask_G3 = discriminator3(multi_class_masking(image, mask[:, 2:3, :, :, :]))
                ad_loss += torch.mean(torch.abs(result3 - mask_G3)) / 3
                # delete
                del result3, mask_G3
                torch.cuda.empty_cache() 
                loss_G = dice_loss + ad_loss
                loss_G.backward()
                optimizer_G.step()

                Losses_G += loss_G.item()
                len_G += 1


            # 保存参数
            if SAVE_PARAMS and (epoch%INTERVAL) == 0:
                state = {'generator':generator.state_dict(), 'discriminator1':discriminator1.state_dict(), 'discriminator2':discriminator2.state_dict(), 'discriminator3':discriminator3.state_dict(), 
                        'GPU':[rank]}
                if not os.path.exists(save_path):
                    os.system(f'mkdir {save_path}')
                torch.save(state, f'{save_path}/epoch_{epoch}_{rank}.pth')


            scheduler_G.step()
            scheduler_D1.step()
            scheduler_D2.step()
            scheduler_D3.step()

            #  validation
            if (epoch%INTERVAL) == 0:
                generator.eval()
                dice, len_val = np.zeros(3), 0
                pr, rc = np.zeros(3), np.zeros(3)
                with torch.no_grad():
                    for i, (val_image, val_mask, index) in enumerate(val_dataloader):
                        val_image,val_mask = val_image.float().cuda(non_blocking=True), val_mask.float().cuda(non_blocking=True)

                        val_out = generator(val_image)[0]     # (N, 3, x, y, z)
                        val_out = torch.sigmoid(val_out)
                        tmp_dice = multi_class_dice_score(val_out.cpu().numpy(), val_mask.cpu().numpy())
                        tmp_pr, tmp_rc = multi_class_precision_and_sensitive(pred=val_out.cpu().numpy(), label=val_mask.cpu().numpy())
                        dice = dice + tmp_dice
                        pr = pr + tmp_pr
                        rc = rc +  tmp_rc
                        len_val += 1
                    # delete
                    del val_out, val_mask, val_image
                    torch.cuda.empty_cache()

                    dice, pr, rc = dice/len_val, pr/len_val, rc/len_val
                    with open(txt_path,'a') as f:
                        f.write(f'epoch:{epoch}\tlocal_rank:{local_rank}\nClass:[WT, TC, ET]\tMean Dice:{dice.mean()}\nVal Dice Score:{dice}\tPrecision:{pr}\tSensitive:{rc}\n')

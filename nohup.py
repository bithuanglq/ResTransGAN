import os



if __name__=='__main__':
    if True:
        # 在 tmux 中运行该程序
        os.system('nohup python -m torch.distributed.launch --nproc_per_node=4 /home/hlq/Project/BraTS/code/ResTransGAN_with_cnn_resnet.py > nohup_v2.out 2>&1 &')

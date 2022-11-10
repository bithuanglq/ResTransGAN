import os



if __name__=='__main__':
    if True:
        # 在 tmux 中运行该程序
        os.system('nohup python -m torch.distributed.launch --nproc_per_node=4 your_python_file_path > nohup.out 2>&1 &')

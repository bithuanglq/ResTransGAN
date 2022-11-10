# ResTransGAN
Core code for " Transformer-based Generative Adversarial Network for Brain Tumor Segmentation"
  
  
  
step 1:  Setup  
```python
pip freeze > requirements.txt
```  

step 2:  Training  
```python
tmux
conda activate your_env_name
python nohup.py
```

step 3:  Inference  
```python
CUDA_VISIBLE_DEVICES=0,1 python code/inference_DP.py
```

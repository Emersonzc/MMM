<p align="center">
  <h1 align="center">MMM: Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?hl=zh-CN&pli=1&user=H1rqfM4AAAAJ" rel="external nofollow noopener" target="_blank"><strong>Jiangming Shi</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=Go9q2jsAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Xiangbo Yin</strong></a>
    ·
    <a href="" target="_blank"><strong>Yeyun Chen</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=a-I8c8EAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yachao Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=CXZciFAAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Zhizhong Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Yuan Xie*</strong></a>    
    ·
    <a href="https://scholar.google.com/citations?user=idiP90sAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Yanyun Qu*</strong></a>       
  </p>
<p align="center">
  <a href="https://arxiv.org/pdf/2401.06825" rel="external nofollow noopener" target="_blank">Access the research paper here</a>

![MMM](imgs/framework.png)
This is an official code implementation of "MMM: Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification", which is accepted by .

## Setup and Train/Test Instructions
```bash
# Install the required packages
pip install torch==1.8.0 torchvision==0.9.1+cu111 faiss-gpu==1.6.3 scikit-learn==1.3.2

# Training Steps for SYSU-MM01

# Step 1: Obtain Features and Pseudo-Labels from Baseline Model
CUDA_VISIBLE_DEVICES=0,1 python Baseline_sysu.py --data-dir dataset_path

# Step 2: Train the MMM Model
CUDA_VISIBLE_DEVICES=0 python main.py --data-dir dataset_path --resume_net1 save_model_name

# Testing for SYSU-MM01
CUDA_VISIBLE_DEVICES=0 python main.py --data-dir dataset_path

```
# Testing Steps for SYSU-MM01 
CUDA_VISIBLE_DEVICES=0 python main.py --data-dir dataset_path



### Contact
jiangming.shi@outlook.com; S_yinxb@163.com.

The code is implemented based on

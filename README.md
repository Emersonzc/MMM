MMM: Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification. (https://arxiv.org/abs/2401.06825)

Environmental requirements:

torch == 1.8.0

torchvision ==  0.9.1+cu111

faiss-gpu  == 1.6.3

scikit-learn == 1.3.2

Training

for SYSU-MM01
Step 1: Obtain Features and Pseudo-Labels from Baseline Model


CUDA_VISIBLE_DEVICES=0,1 python Baseline_sysu.py --data-dir dataset_path

Step 2: Train the MMM Model


CUDA_VISIBLE_DEVICES=0 python main.py --data-dir dataset_path --resume_net1 save_model_name


```
### Testing

# for SYSU-MM01
CUDA_VISIBLE_DEVICES=0 python main.py --data-dir dataset_path



### Contact
jiangming.shi@outlook.com; S_yinxb@163.com.

The code is implemented based on

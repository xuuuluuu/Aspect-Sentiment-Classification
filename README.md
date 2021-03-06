# ABSA
[EMNLP 2020] [Aspect Sentiment Classification with Aspect-Specific Opinion Spans](https://arxiv.org/abs/2010.02696)

# Requirement  
Python 3.7.3  
Pytorch 1.1.0  
Numpy 1.17.2  
CUDA: 10.0  

Suggested Environment Installation Procedures:  
conda create -n yourenvname python=3.7 anaconda  
conda install -n yourenvname pytorch==1.1.0 torchvision cudatoolkit=10.0 -c pytorch  

# Running   
```
python train_crf_glove.py  
```
By default, the model runs on laptop dataset with provided hyper-parameters.  
To run on other datasets:  
  - In line 131-144, uncomment the main funtion for different dataset.  
  - Edit the embed_num on line 148 and path directory in line 171-176, .

To run with the saved best model:  
  - Set training to False on line 147.

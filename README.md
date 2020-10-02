# ABSA

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
  - comment lines 184-192 and uncomment the main funtion for the dataset you want to run.  
  - edit the embed_num on line 199 and path directory on line 223-230  

To run with the saved best model:  
  - Set training to False on line 197.

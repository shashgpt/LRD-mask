# A Mask-based Logic Rules Dissemination Method for Sentiment Classifiers
This repository contains the code and dataset used in our paper accepted for publication at ECIR2023

## Covid-19 twitter dataset for testing Logic Rule Dissemination
We constructed a dataset from tweets based on Covid-19 topic for sentence-level binary sentiment classification task. The dataset can be downloaded from this google-drive link: https://drive.google.com/file/d/1p3yZ-L8OxsZYuXOJoOn8aspYFLnn68b_/view?usp=share_link

## Requirements
1) Anaconda 4.9.2 environment with Python 3.8.12
2) Specific packages to be installed in the environment as mentioned in requirements.txt
3) Ubuntu 20.04

## Instructions to run the code and reproduce results

1) Download the dataset from the Google Drive link and place it in the covid_19_tweets_dataset folder
2) To reproduce results for base models, specify the configuration parameters in base_model/config/config.py and run base_model/main.py
3) To reproduce results for IKD models, specify the configuration parameters in IKD/config/config.py and run IKD/main.py
4) To reproduce results for CWE models 
   - Install docker (version 20.10.12, build e91ed57)
   - Install Nvidia Docker toolkit for GPU support in Docker (https://github.com/NVIDIA/nvidia-docker/blob/master/README.md#quickstart).
   - Run "sudo docker run --mount type=bind,source="$(pwd)",target=/mnt --gpus all -it --rm tensorflow/tensorflow:1.7.0-gpu-py3 bash"
   - Run "pip install keras==2.2.0 tensorflow_hub==0.1.1 tqdm lime"
   - Navigate to /mnt directory
   - Specify the configuration parameters in CWE/config/config.py
   - Run CWE/main.py
6) To reproduce results for mask models, specify the configuration parameters in mask_model/config/config.py and run mask_model/main.py
7) To view the results presented in the paper, run analysis.ipynb

# Residual-learning-based-two-stream-network-for-RGB-T-object-tracking
This repository contains the codes for paper "**Residual learning-based two-stream network for RGB-T object tracking**" by Yili Chen, Minjie Wan*, Yunkai Xu, et al. (*Corresponding author).

The overall repository style is partially borrowed from MANet (https://github.com/Alexadlu/MANet). Thanks to Chenglong Li.

The relative datasets and evaluation toolkits can be downloded from the following links:
## RGBT234 dataset
Link：https://pan.baidu.com/s/1weaiBh0_yH2BQni5eTxHgg 
code：qvsq
## RGBT210 dataset
Link：https://pan.baidu.com/s/1FClmX0SH3WarcczkEQbmwA 
code：ps8j 
## GTOT dataset
Link：https://pan.baidu.com/s/1zaR6aXh9PVQs063Q_b9zQg 
code：ajma
## RGBT234 toolkit
Link：https://pan.baidu.com/s/1UksOGtD2yl6k8mtB-Wr39A 
code：4f68
## RGBT210 toolkit
Link：https://pan.baidu.com/s/1KHMlbhu5R29CJvundGL4Sw 
code：8wtc
## GTOT toolkit
Link：https://pan.baidu.com/s/1iVVAXS4LZLvoQSGQnz7ROw 
code：d53m

## Requierments:

* python2.7
* pytorch == 0.3.1
* numpy
* PIL
* by yourself need install some library functions 

## Usage
1. Generate train data using .../pretrain/prepro_data.py.
2. Train your own model using .../pretrain/train.py, the relative parameters can be adjusted in option.py and train.py.
3. Track with the trained model by running .../tracking/run_tracker.py, some parameters need to be set in .../tracking/options.py and track.py.



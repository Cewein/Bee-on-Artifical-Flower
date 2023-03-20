#!/bin/bash

#load submodule, i.e yolov7
git submodule init
git submodule update

#download the dataset
d='./baf/' # unzip directory
url=https://app.roboflow.com/ds/9h5VNVqe1R?key=0kT6Sm7Cxw
f='BAF.v4-default.yolov7pytorch.zip'

echo 'downloading' $f
curl -L $url -o $f && unzip -q $f -d $d && rm $f

#go into the dataset for training
cd yolov7

p='p5'
cy='../training/tl-training.yaml'
cyw6='../training/yolov7-w6-custom.yaml'
cydf='../training/yolov7-custom.yaml'

#choose between two type of training
if [$p=='p5']; then
    wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    python train.py --workers 8 --device 0 --batch-size 32 --data $cy --img 640 640 --cfg $cydf --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
else
    wget -nc https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt
    python train_aux.py --workers 8 --device 0 --batch-size 16 --data $cy --img 1280 1280 --cfg $cyw6 --weights 'yolov7-w6_training.pt' --name yolov7-w6-custom --hyp data/hyp.scratch.custom.yaml
fi
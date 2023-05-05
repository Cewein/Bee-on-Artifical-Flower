#!/bin/bash

# Load submodule, i.e. yolov7
git submodule init
git submodule update

# Download the dataset
DATASET_DIR='./baf/'

if [ -d "$DATASET_DIR" ]; then
    echo "Directory $DATASET_DIR exists. Skipping database download." 
else
    URL='https://app.roboflow.com/ds/4TadDrWzxR?key=YBmMXGBZ62'
    FILENAME='BAF.yolov7pytorch.zip'

    echo "Downloading $FILENAME..."
    curl -L "$URL" -o "$FILENAME" && unzip -q "$FILENAME" -d "$DATASET_DIR" && rm "$FILENAME"
fi

# Go into the yolov7 repo for training
cd yolov7

# Training parameters
TRAINING_WORKERS=4
TRAINING_DEVICE=0
TRAINING_BATCH_SIZE=16

# Custom YOLO configurations
YOLO_CUSTOM='../training/yolov7-custom.yaml'
YOLO_W6_CUSTOM='../training/yolov7-w6-custom.yaml'
YOLO_HYP='../training/hyp.scratch.custom.yaml'

# Choose between two types of training
MODEL_WEIGHTS=""
MODEL_CFG=""
MODEL_NAME=""
IMG_SIZE=""
if [ "$1" == "p5" ]; then
    MODEL_WEIGHTS='yolov7_training.pt'
    MODEL_CFG="$YOLO_CUSTOM"
    MODEL_NAME='yolov7-custom'
    IMG_SIZE="640 640"
elif [ "$1" == "p6" ]; then
    MODEL_WEIGHTS='yolov7-w6_training.pt'
    MODEL_CFG="$YOLO_W6_CUSTOM"
    MODEL_NAME='yolov7-w6-custom'
    IMG_SIZE="1280 1280"
else
    echo "Please provide the model type to train (either 'p5' or 'p6') as the first argument."
    exit 1
fi

# Download the appropriate model weights
wget -nc "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/$MODEL_WEIGHTS"

# Train the model
if [ "$1" == "p5" ]; then
    python train.py --workers "$TRAINING_WORKERS" --device "$TRAINING_DEVICE" --batch-size "$TRAINING_BATCH_SIZE" --data "$YOLO_CUSTOM" --img "$IMG_SIZE" --cfg "$MODEL_CFG" --weights "$MODEL_WEIGHTS" --name "$MODEL_NAME" --hyp "$YOLO_HYP"
else
    python train_aux.py --workers "$TRAINING_WORKERS" --device "$TRAINING_DEVICE" --batch-size "$TRAINING_BATCH_SIZE" --data "$YOLO_CUSTOM" --img "$IMG_SIZE" --cfg "$MODEL_CFG" --weights "$MODEL_WEIGHTS" --name "$MODEL_NAME" --hyp "$YOLO_HYP"
fi

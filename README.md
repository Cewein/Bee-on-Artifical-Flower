# B.A.F (Bee on Artifical Flower) 

## Table of Contents

- [Description](#description)
- [Dataset](#dataset)
- [Example](#example)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [YoloV7](#yolov7)
- [License](#license)

## Description

This project is a collection of Python scripts designed to detect and track bees in videos. The scripts use YOLOv7 object detection model to detect the bees and flowers in the videos. The project also includes an example script which shows how to use the other scripts to detect and track bees in a video. 

## Dataset

To be done.

## Example

The example directory contains example scripts that demonstrate how to use the project scripts to detect and track bees in a video. The example scripts include `BeeDetection.py`, `BoudingQuadDemo.py`, and `videoDemo.py` and more. 

## Installation

To install the project, clone the repository from GitHub:

```
git clone https://github.com/cewein/Bee-on-Artifical-Flower.git
git submodule init
git submodule update
```

Sumodules are need, YOLOv7 being one.

Then, navigate to the root directory of the project and install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To be done.

## Training

The training directory contains scripts and configuration files for training the YOLOv7 object detection model. The `train.sh` or `train.ps1` script can be used to train the model. 

there is two mode for the traning: **p5** and **p6**. They used different version of YOLOv7, p5 is the default, p6 is q heavier version and migth not properly run.

the training use transfer learning as a methode of training and can be bound to any existing database. here is only for a specific case.

run this script from the root folder.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
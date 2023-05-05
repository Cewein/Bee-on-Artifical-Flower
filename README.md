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

This project is a collection of Python scripts designed to detect and track bees in videos. The scripts use YOLOv7 object detection model to detect the bees in the videos, markov quadrilateral algorithm to track their movement, and OpenCV to process the videos. The project also includes an example script which shows how to use the other scripts to detect and track bees in a video. 

## Dataset

To be done.

## Example

The example directory contains example scripts that demonstrate how to use the project scripts to detect and track bees in a video. The example scripts include `BeeDetection.py`, `BoudingQuadDemo.py`, and `videoDemo.py` and more. 

## Installation

To install the project, clone the repository from GitHub:

```
git clone https://github.com/cewein/Bee-on-Artifical-Flower.git
```

Then, navigate to the root directory of the project and install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

To use the project, navigate to the root directory of the project and run the desired script. For example, to detect and track bees in a video, run the `BeeDetection.py` script:

```
python example/BeeDetection.py
```

## Training

The training directory contains scripts and configuration files for training the YOLOv7 object detection model. The `train.sh` script can be used to train the model. 

## YoloV7

The yolov7 directory contains the configuration files and data required to use the YOLOv7 object detection model. The cfg subdirectory contains the configuration files for the YOLOv7 model. The data subdirectory contains the configuration files for the training data used to train the YOLOv7 model. 

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
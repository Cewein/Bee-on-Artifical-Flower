import subprocess

import numpy as np

# Run YOLOv7 detection
def runYolov7Detection(weightPath, dataPath, savePath):
    cmdStr = f"cd yolov7/ && python3 detect.py --weights {weightPath} --project ../{savePath} --save-txt --conf 0.20 --img-size 640 --source {dataPath}"
    subprocess.run(cmdStr, shell=True)

import os
import subprocess

def runYolov7Detection(weightPath, dataPath, savePath, confThreshold=0.20, imgSize=640, extraArgs=None):
    # Validate input arguments
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weightPath. File does not exist.")
    if not os.path.exists(dataPath):
        raise ValueError("Invalid dataPath. File or directory does not exist.")
    if extraArgs is None:
        extraArgs = []

    # Change the working directory to yolov7
    yolov7Dir = "yolov7/"
    os.chdir(yolov7Dir)

    # Construct the command and arguments as a list of strings
    cmd = ["python3","detect.py","--weights",weightPath,"--project",f"../{savePath}","--save-txt",
        "--conf",str(confThreshold),"--img-size",str(imgSize),"--source",dataPath]

    # Add additional command-line arguments if provided
    cmd.extend(extraArgs)

    try:
        # Run the YOLOv7 detection command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running YOLOv7 detection: {e}")

    # Change the working directory back to the previous directory
    os.chdir("..")

def loadResult(path: str):
    result = np.loadtxt(path)

    ids = np.int32(result[:,0])
    bboxs = result[:,1:]

    return ids, bboxs

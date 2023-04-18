# %%  imports
%load_ext autoreload
%autoreload 2

#local
import src.data.video as video
import src.geometry.boundingBox as BBox
import src.geometry.geometry as geometry
import src.geometry.markovQBox as QBox
import src.yolo.yolo as yolo

#os lib
import subprocess
import os
import shutil

#math
import matplotlib.pyplot as plt
import numpy as np
import scipy

#video processing openCV
import cv2 as cv
from skimage import transform

# %%
videoPath = "dataset/video/MAH00031.MP4"
frameIndex = 200 #set to -1 for full

img = video.openFrame(videoPath, frameIndex)
rois = yolo.detectFlowers(img)

# %%
boundingQuad = QBox.boudingQuadFromROI(rois)

#get the homographic perspective change
tform3 = geometry.getPerspectiveTransform(boundingQuad)

# %%

# Initialize video capture object with specified path
cap = cv.VideoCapture(videoPath)

# Read first frame of video and initialize HSV image with maximum saturation
ret, prevFrame = cap.read()
hsv = np.zeros_like(img, dtype=np.uint8)
hsv[..., 1] = 255

# Initialize optical flow variable
flow = None

# Loop through each frame in the video
while cap.isOpened():
    # Read the next frame of the video
    ret, frame = cap.read()

    # If the frame was not read correctly, break out of the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert current and previous frames to grayscale and warp them
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayPrev = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    # gray = transform.warp(gray, tform3, output_shape=(500, 500))
    # grayPrev = transform.warp(grayPrev, tform3, output_shape=(500, 500))

    # Calculate optical flow using Farneback algorithm
    flow = cv.calcOpticalFlowFarneback(grayPrev, gray, flow, 0.5, 5, 5, 5, 6, 1.0, 0)

    # Convert flow vectors to polar coordinates and extract angle and magnitude information
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert angle and magnitude information to hue and value channels, respectively, in HSV image
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV image to BGR format for display
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # Display flow visualization and original frame
    cv.imshow('Optical Flow', bgr)
    cv.imshow('Original Frame', gray)

    # Wait for key press event and exit loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

    # Store current frame as previous frame for next iteration of loop
    prevFrame = frame

# Release video capture object and close all windows
cap.release()
cv.destroyAllWindows()

# %%

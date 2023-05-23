import argparse

import sys
sys.path.append('./src/python')

#local
import utils
import geometry.boundingBox as BBox
import geometry.geometry as geometry
import geometry.markovQBox as QBox
import yolo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect fake flowers with YOLOv7.")

    parser.add_argument("--dataPath", type=str, help="path to video or image file")
    parser.add_argument("--frameIndex", type=int, help="index of frame to process", default=None)
    parser.add_argument("--weightFlower", type=str, help="path to YOLOv7 weights file", default="../training/result/best-fake-flower.pt")
    parser.add_argument("--weightBee", type=str, help="path to YOLOv7 weights file", default="../training/result/best-bee-custom-v2.pt")
    parser.add_argument("--tmpDir", type=str, help="path to YOLOv7 weights file", default="tmp/")
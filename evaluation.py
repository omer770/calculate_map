from matplotlib import pyplot as plt
import json
import os
import numpy as np
import cv2

annFile='ml-datasets/dataset/yolo-data/annotations/val.json'
imageDir = '/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/images/val/'


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
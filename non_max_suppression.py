import os
from matplotlib import pyplot as plt
import json
import numpy as np
import cv2
#from pycocotools import mask as mask
from skimage import draw
from pycocotools import mask
from skimage import measure
from itertools import groupby
from shapely import geometry
import pandas as pd 

def generate_poly_pred(segmentation_list):
  if type(segmentation_list) == str:
    segmentation_list = segmentation_list.replace('[','').replace(']','').split(',')
    segmentaionlist3 = []
    for p in range(0,len(segmentation_list),2):
      segmentaionlist3.append([int(segmentation_list[p]),int(segmentation_list[p+1])])
    segmentation_list =segmentaionlist3
    if segmentation_list[0] != segmentation_list[-1]:
      if (abs(segmentation_list[0][0]-segmentation_list[-1][0]) <=7 )&(abs(segmentation_list[0][1]-segmentation_list[-1][1]) <=7):
        segmentation_list[-1] = segmentation_list[0]
      else:
        segmentation_list.append(segmentation_list[0])
  return geometry.Polygon(segmentation_list)

def calculate_iou(pts1,pts2):
  poly1 = generate_poly_pred(pts1)
  poly2 = generate_poly_pred(pts2)
  return poly1.intersection(poly2).area/poly1.union(poly2).area

def non_max_sup(pred_object,threshold,nms_threshold ):

  remove_dict = {}
  fname_list = pred_object.keys()
  #fname = '41346_89723_18.jpg'
  for fname in fname_list:
    #pred_object[fname]
    conf = []
    cls = []
    seg = []
    s = list(pred_object[fname].keys())
    for i in range(len(s)):
      conf.append(pred_object[fname][s[i]]['confidence'])
      cls.append(pred_object[fname][s[i]]['classname'])
      seg.append(pred_object[fname][s[i]]['segmentation'])
    df_sorted = pd.DataFrame(zip(conf,cls,seg),columns = ['confidence','classname','segmentation']).sort_values(by=['confidence'], ascending=False)
    df_sorted = df_sorted[df_sorted['confidence'] >= threshold]
    df_sorted1 = df_sorted[df_sorted['confidence'] < threshold]
    for l in range(len(df_sorted)):
      for m in range(l+1,len(df_sorted)):
        iou_nms  = calculate_iou(df_sorted.segmentation[l],df_sorted.segmentation[m])
        #iou_nms = 
        if iou_nms >= nms_threshold:
          print(fname,l,m,s[m],iou_nms)
          sub_dict = pred_object[fname]
          try:del sub_dict[s[m]]
          except: pass
          remove_dict[fname] = s[m]
  return(remove_dict)

if __name__ == '__main__':
  pass

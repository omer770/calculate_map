import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from shapely import geometry
refp,refr = 0.0001,0.0001
#from coco_to_csv_converter import coco_to_csv

def main():
  pass

def generate_poly_from_list(segmentation_list):
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

def precision_recall_per_object(tp_fp,len_gt, idr = 0,clas = 0):
  #file_index
  # 1 for pool, 0 for building
  precision = (np.sum(tp_fp[idr][clas]))/(len(tp_fp[idr][clas])) if (len(tp_fp[idr][clas])) != 0 else refp
  recall = (np.sum(tp_fp[idr][clas]))/(len_gt[idr][clas]) if (len_gt[idr][clas]) != 0 else refr
  return precision,recall

def overall_PR(tp_fp,len_gt,classes):
  pc = 0
  rc = 0
  count = 0
  print("-"*43)
  count_ngt = 0
  
  for c in range(len(classes)):
    prc = 0
    rcc = 0
    count_ngt1 = 0
    for i in range(len(len_gt)):
      pri, rci = precision_recall_per_object(tp_fp,len_gt, i,c)
      if (pri, rci)==(refp, refr):
        count_ngt1+=1
      prc += pri
      rcc += rci
    #print("nt1",count_ngt1)
    prcn = prc /(len(tp_fp)-count_ngt1)
    rccn = prc /(len(len_gt)-count_ngt1)

    count_ngt+=count_ngt1
    print("precision for ",classes[c],prcn)
    print("recall for ",classes[c],rccn)
    print("-"*43)
    pc+=prc
    rc+=rcc
  pc/=(len(tp_fp)*2-count_ngt)
  rc/=(len(tp_fp)*2-count_ngt)
  print("-"*43)
  print("The overall precision is ",pc)
  print("The overall recall is ",rc)
  print("-"*43)


def calculate_iou(pts1,pts2):
  poly1 = generate_poly_from_list(pts1)
  poly2 = generate_poly_from_list(pts2)
  return poly1.intersection(poly2).area/poly1.union(poly2).area

def iou_metric_calculate(pred_object,df_gt,iou_threshold):
  p_r = {}
  polygons = []
  tp_fp = []
  len_gt= []
  polygons1 = []
  fnames = df_gt['File Name'].unique()
  for fname in fnames:
    print("File Name- ",fname)
    print("-"*85)
    if len(pred_object[fname])==0:
      continue
    tp_fp_1 = []
    len_gt1 = []
    for cls in df_gt['Class Name'].unique():
      p_r[cls]= []
      print("Class- ",cls)
      s = list(pred_object[fname].keys())
      t_f= []
      len_gt2 = []
      count_np = 0
      count_gt = 0
      for t in range(len(s)):
        #print("Row- ",s[t])
        #print("confidence- ",pred_object[fname][s[t]]['confidence'])
        seg1 = pred_object[fname][s[t]]['segmentation']
        sub_df = df_gt[((df_gt['File Name']== fname )&(df_gt['Class Name']== cls))]
        idx = sub_df.Segmentation.index
        if pred_object[fname][s[t]]['classname'] != cls:
          count_np+=1
          continue
        for id in range(len(idx)):
          if len(sub_df) == 0:
            continue
          seg2 = sub_df.Segmentation[idx[id]]
          try:iou = calculate_iou(seg1,seg2)
          except:
            polygons.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
            iou = 0.1
            #print("error at ",idx[id])
          #if iou>0.1:print("row ,ID and iou- ",s[t],idx[id],iou)
          if iou >= iou_threshold:
            #print("iou",s[t],id, iou,idx[id] ,"TP")
            t_f.append(1)
            break
          else:
            if (id==len(idx)-1):
              t_f.append(0)
              polygons1.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
              #print('FP')
        len_gt2.append(len(sub_df))
        if np.sum(t_f)==len(sub_df):
            break
      print("t_f, sum, len ",t_f,np.sum(t_f),len(t_f))
      if len(t_f) != 0:
        print("len pred ",len(s))
        print("len of sub_df ",len(sub_df))
        pre = (np.sum(t_f))/(len(t_f)) if (len(t_f))!= 0 else refp
        rec = (np.sum(t_f))/(len(sub_df)) if (len(sub_df))!=0 else refr
        print()
        print("Precision and Recall for class- ",cls,pre,rec)
      tp_fp_1.append(t_f)
      len_gt1.append(len(sub_df))
      print("len pred ",len(s)-count_np)
      print("len of gt ",len(sub_df))
      print("-"*85)
    tp_fp.append(tp_fp_1)
    len_gt.append(len_gt1)

  return tp_fp,len_gt





if __name__ == "__main__":
  main()
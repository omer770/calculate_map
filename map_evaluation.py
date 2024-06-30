import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from shapely import geometry
from sklearn.metrics import auc
import os

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
  precision = (np.sum(tp_fp[idr][clas]))/(len(tp_fp[idr][clas])) if (len(tp_fp[idr][clas])) != 0 else 0.0001
  recall = (np.sum(tp_fp[idr][clas]))/(len_gt[idr][clas]) if (len_gt[idr][clas]) != 0 else 0.0001
  return precision,recall

def overall_PR(tp_fp,len_gt,classes):
  pc = 0
  rc = 0
  count = 0
  print("-"*43)
  count_ngt = 0
  #classes = ['building','pool']
  for c in [0,1]:
    prc = 0
    rcc = 0
    count_ngt1 = 0
    for i in range(len(len_gt)):
      pri, rci = precision_recall_per_object(tp_fp,len_gt, i,c)
      if (pri, rci)==(0.0001, 0.0001):
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
  return None


def calculate_iou(pts1,pts2):
  poly1 = generate_poly_from_list(pts1)
  poly2 = generate_poly_from_list(pts2)
  return poly1.intersection(poly2).area/poly1.union(poly2).area

def perform_pr_process(df_apc):
  Acc_TP = [] 
  Acc_FP = []
  Precision_ap = []
  Recall_ap = []
  count_acc_tp =  0
  count_acc_fp = 0
  #df_apc = df_apc.sort_values(by=['Confidence'])
  len_gt_subclass = len(df_apc)
  for row_T,row_F in df_apc.loc[:,["TP","FP"]].values:
    count_acc_tp += row_T
    count_acc_fp += row_F
    Acc_TP.append(count_acc_tp)
    Acc_FP.append(count_acc_fp)
    presn = (count_acc_tp)/(count_acc_tp+count_acc_fp) if (count_acc_tp+count_acc_fp)!= 0 else 0
    rcll = (count_acc_tp)/(len_gt_subclass) if len_gt_subclass!= 0 else 0
    Precision_ap.append(presn)
    Recall_ap.append(rcll)
  df_apc["Acc_TP"] = Acc_TP
  df_apc["Acc_FP"] = Acc_FP
  df_apc["Precision_ap"] = Precision_ap
  df_apc["Recall_ap"] = Recall_ap


def map_calculate_per_iou(pred_object,df_gt,iou_threshold,threshold):
  tp_fp,len_gt,polygons1,polygons =[],[],[],[]
  classes = df_gt['Class Name'].unique()
  fnames = df_gt['File Name'].unique()
  FileName_list,Object_list,Confidence_list,TP_list,FP_list,Class_list = [],[],[],[],[],[]
  df_ap_class = {}
  for fname in fnames:
    count_gt_class = 0
    print(f"File Name- {fname}, iou_threshold- {iou_threshold}")
    print("-"*85)
    if len(pred_object[fname])==0:continue
    tp_fp_1, len_gt1 =[],[]
    for cls in classes:    
      print("Class- ",cls)
      s = list(pred_object[fname].keys())
      t_f =[]
      len_gt2 = []
      count_np = 0
      sub_df = df_gt[((df_gt['File Name']== fname )&(df_gt['Class Name']== cls))]
      idx = sub_df.Segmentation.index
      for t in range(len(s)):
        if pred_object[fname][s[t]]['confidence']<threshold:  continue
        seg1 = pred_object[fname][s[t]]['segmentation']
        if pred_object[fname][s[t]]['classname'] != cls:
          count_np+=1
          continue
        for id in range(len(idx)):
          if len(sub_df) == 0:continue
          seg2 = sub_df.Segmentation[idx[id]]
          try:iou = calculate_iou(seg1,seg2)
          except:
            polygons.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
            iou = 0.1
          if iou >= iou_threshold:
            t_f.append(1)
            FileName_list.append(fname)
            Object_list.append(s[t])
            Confidence_list.append(pred_object[fname][s[t]]['confidence'])
            Class_list.append(cls)
            TP_list.append(1)
            FP_list.append(0)
            break
          else:
            if (id==len(idx)-1):
              t_f.append(0)
              #print("t_f ",t_f)
              FileName_list.append(fname)
              Object_list.append(s[t])
              Confidence_list.append(pred_object[fname][s[t]]['confidence'])
              TP_list.append(0)
              FP_list.append(1)
              Class_list.append(cls)
              polygons1.append([generate_poly_from_list(seg1),generate_poly_from_list(seg2)])
              #print('FP')
        len_gt2.append(len(sub_df))
        if np.sum(t_f)==len(sub_df):
            break
      print("t_f, sum, len ",t_f,np.sum(t_f),len(t_f))
      if len(t_f) != 0:
        print("len pred ",len(s))
        print("len of gt ",len(sub_df))
        pre = (np.sum(t_f))/(len(t_f)) if (len(t_f))!= 0 else 0.0001
        rec = (np.sum(t_f))/(len(sub_df)) if (len(sub_df))!=0 else 0.0001
        print()
        print("Precision and Recall for class- ",cls,pre,rec)
      tp_fp_1.append(t_f)
      len_gt1.append(len(sub_df))
      #count_gt_class +=len(sub_df)
      
      print("len pred ",len(s)-count_np)
      print("len of gt ",len(sub_df))
      print("-"*85)
    tp_fp.append(tp_fp_1)
    len_gt.append(len_gt1)

  df_ap = pd.DataFrame(zip(FileName_list,Object_list,Confidence_list,Class_list,TP_list,FP_list),columns = ["Image","Detection", "Confidence", "Class","TP","FP"])
  df_ap = df_ap.sort_values(by=['Confidence'],ascending=False)
  for cls in classes:
    df_ap_class[cls] = df_ap[df_ap['Class'] == cls]
    df_apc = df_ap_class[cls]
    perform_pr_process(df_apc)

  return tp_fp,len_gt,df_ap_class

def calculate_maP(df_ap_class,classes,iou_threshold):
  aP = {}
  ap_df = {}
  for cls in classes:
    df_ap_graph = df_ap_class[cls].copy()
    aP[cls] = auc(df_ap_graph['Recall_ap'],df_ap_graph['Precision_ap'])
    ap_df[cls] = df_ap_graph.loc[:,['Recall_ap','Precision_ap']]
  print("The aP of the model ",aP)
  maP = np.sum(list(aP.values()))/len(classes)
  print("The maP of the model ",maP)
  return maP,aP,ap_df

if __name__ == "__main__":
  main()
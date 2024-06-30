import json
import os
import numpy as np
import cv2
import csv
from pycocotools.coco import COCO

def main():
  pass

def create_dicts_from_img_ids(image_ids,coco_json_file):
  id_2_names = {}
  names_2_ids = {}
  coco=COCO(coco_json_file)
  for img_id in image_ids:
    id_2_names[img_id] = coco.loadImgs(img_id)[0]['file_name']
    names_2_ids[coco.loadImgs(img_id)[0]['file_name']]=img_id
  return id_2_names,names_2_ids

def coco_to_csv(coco_json_file,coco_csv_path):
  # Opening JSON file
  with open(coco_json_file, 'r') as openfile:
    # Reading from json file
    coco_object = json.load(openfile)
  coco=COCO(coco_json_file)
  image_ids = coco.getImgIds()
  image_info = coco.loadImgs(image_ids[0])[0]
  height, width = image_info['height'], image_info['width']
  id_2_names,names_2_ids = create_dicts_from_img_ids(image_ids,coco_json_file)
  classname_2_category_id = {}
  category_id_2_classname = {}
  cats = coco_object['categories'].copy()
  for l in range(len(cats)):
    category_id_2_classname[cats[l]['id']] = cats[l]['name']
    classname_2_category_id[cats[l]['name']] = cats[l]['id']
  with open(coco_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header row in the CSV file
    csvwriter.writerow(["File Name", "Class Name", "Segmentation"])
    for i in range(len(coco_object['annotations'])):
        seg = coco_object['annotations'][i]['segmentation'][0]
        if seg ==[]:
          continue
        seg1 =[]
        for j in range(0,len(seg),2):
          seg1.append([int(seg[j]),int(seg[j+1])])

        csvwriter.writerow([id_2_names[coco_object['annotations'][i]['image_id']],
         category_id_2_classname[coco_object['annotations'][i]['category_id']],seg1])

  return None


if __name__ == "__main__":
  main()
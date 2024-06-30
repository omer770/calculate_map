from matplotlib import pyplot as plt
import numpy as np
from shapely import geometry
from shapely.geometry import  MultiPolygon

def generate_poly(segmentation_list):
  if type(segmentation_list) == str:
    segmentation_list = segmentation_list.replace('[','').replace(']','').split(',')
  pointly = []
  for i in range(0,len(segmentation_list),2):
      pointly.append([int(segmentation_list[i]),int(segmentation_list[i+1])])
  #print(f"Expected mask sum: {expected_mask.sum()}")
  poly = geometry.Polygon(pointly)
  return geometry.Polygon(pointly)

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

def plt_poly(polygon,shape):
  fig, ax = plt.subplots()
  for poly in polygon.geoms:
      xe, ye = poly.exterior.xy
      #xe, ye = [1024,1024]
      ax.plot(xe, ye, color="blue")
  plt.ylim(0, shape[1])
  plt.xlim(0, shape[0])
  plt.show()

def get_multi_poly_fname(fname, pred_object,df_gt):
  classes = df_gt['Class Name'].unique()
  seg1_polylist =[]
  seg2_polylist =[]
  colors = ["green","blue"]
  c1 = []
  c2 = []
  for i in range(len(classes)):
    #c1[i] = colors[i]#colors[i]
    #c2[i] = 
    s = list(pred_object[fname].keys())
    for t in s:
      if pred_object[fname][t]['classname']==classes[i]:
        seg1_polylist.append(generate_poly_from_list(pred_object[fname][t]['segmentation']))
        c1.append(colors[i])
    sub_df = df_gt[(df_gt['File Name']== fname)&(df_gt['Class Name']== classes[i])]
    for seg in sub_df.Segmentation:
      seg2_polylist.append(generate_poly_from_list(seg))
      c2.append(colors[i])
  return seg1_polylist,seg2_polylist,c1,c2

def plt_multi_poly_lists(polygons1, polygons2,c1,c2,shape):
  fig, axs = plt.subplots(1, 2)
  for color,poly in zip(c1,MultiPolygon(polygons1).geoms ):
    #Mpoly = (poly)
    xe, ye = poly.exterior.xy
    axs[0].plot(xe, ye, color=color)
  axs[0].set_title('Predictions')
  axs[0].set_ylim(0, shape[1])
  axs[0].set_xlim(0, shape[0])

  for color,poly  in zip(c2,MultiPolygon(polygons2).geoms):
    #Mpoly = (poly) 
    xe, ye = poly.exterior.xy
    axs[1].plot(xe, ye, color=color)
  axs[1].set_title('Annotations')
  axs[1].set_ylim(0, shape[1])
  axs[1].set_xlim(0, shape[0])

  plt.show()

def plt_multi_poly(polygon,shape):
  fig, ax = plt.subplots()
  for poly in MultiPolygon(polygon).geoms:
      xe, ye = poly.exterior.xy
      #xe, ye = [1024,1024]
      ax.plot(xe, ye, color="blue")
  plt.ylim(0, shape[1])
  plt.xlim(0, shape[0])
  plt.show()


if __name__ == '__main__':

  from matplotlib import pyplot as plt
  import numpy as np
  from shapely import geometry

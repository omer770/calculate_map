import json
import os
import numpy as np
import cv2



def main():

  with open('Benchmarking.json', 'r') as openfile: 
      # Reading from json file
      json_object = json.load(openfile)

  image_ids = list(json_object.keys())
  path_2_out = '/home/ubuntu/workspace/new_dataset/out/'
  brown = (165,42,42)
  blue = (65,105,225)
  for i in range(5):
      #for i in range(len(image_ids)):
      id_i = json_object[image_ids[i]]
      img = cv2.imread(os.path.join( "/home/ubuntu/workspace/new_dataset/test/images",image_ids[i]))
      for k in id_i.keys():
          # Polygon corner points coordinates
          pts = np.array(id_i[k]['segmentation'], np.int32)
          pts = pts.reshape((-1, 1, 2))
          isClosed = True 

          # Line thickness of 2 px
          color = brown
          thickness = 2
          # Using cv2.polylines() method
          # Draw a Blue polygon with 
          # thickness of 1 px
          if id_i[k]['classname'] == 'pool':
              color = blue
          img = cv2.polylines(img, [pts], 
                              isClosed, color, thickness)
      cv2.imwrite(path_2_out+'image_'+image_ids[i],img)
      print('Saved: image_'+image_ids[i])
  print("Saved at location: "+ )





if __name__ == "__main__":
  main()
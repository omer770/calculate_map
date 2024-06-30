from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import json
import os

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    my_new_model = YOLO('/home/ubuntu/workspace/new_dataset/results/20_epochs-5/weights/last.pt')

    dir_path_test = '/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/images/val/'
    class_names = ['building', 'pool']
    lst = os.listdir(dir_path_test)
    dict_i = {}
    for i in range(len(lst)):
        new_results = my_new_model.predict(dir_path_test+lst[i] , conf=0.2 )
        bbox = new_results[0].boxes.data[:,:-1]
        cls_id = new_results[0].boxes.data[:,-1]
        cls_rows = [class_names[int(x)] for x in cls_id ]
        confidnce = list(new_results[0].boxes.conf.cpu().numpy())
        dict_j = {}
        for j in range(len(cls_rows)):
            dict_k = {}
            dict_k['segmentation'] = new_results[0].masks.xy[j]
            #dict_k['bbox'] = bbox[j]
            dict_k['classname'] = cls_rows[j]
            dict_k['confidence'] = confidnce[j]
            dict_j[j] = dict_k
        dict_i[lst[i]] = dict_j
        
    dumped = json.dumps(dict_i, cls=NumpyEncoder)
    with open('Benchmarking.json', 'a') as f:
        f.write(dumped + '\n') 

    

if __name__ == "__main__":
  main()
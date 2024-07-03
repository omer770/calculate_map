import json
import numpy as np

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
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)
def print_structure(d, indent=0):
    """Print the structure of a dictionary or list."""

    # If the input is a dictionary
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key))
            print_structure(value, indent+1)

    # If the input is a list
    elif isinstance(d, list):
        print('  ' * indent + "[List of length {} containing:]".format(len(d)))
        if d:
            print_structure(d[0], indent+1)
          
def retrive_json_object(json_file):
  assert json_file.split('.')[-1] == 'json' , f"Provide a Json extention filename, provided-'{json_file}' "
  with open(json_file, 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)
  return json_object

def export_json_file(json_object,filename):
  assert filename.split('.')[-1] == 'json' , f"Provide a Json extention filename, provided-'{filename}' "
  dumped = json.dumps(filename, cls=NumpyEncoder)
  with open(filename, 'a') as f:
      f.write(dumped + '\n')
  print("Saved- ",filename)
  return filename
  

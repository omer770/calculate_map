import json
import os
import shutil
import yaml

# Function to convert images to YOLO format
def convert_to_yolo(input_images_path, input_json_path, output_images_path, output_labels_path):
    # Open JSON file containing image annotations
    f = open(input_json_path)
    data = json.load(f)
    f.close()

    # Create directories for output images and labels
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # List to store filenames
    file_names = []
    for filename in os.listdir(input_images_path):
        if filename.endswith(".jpg"):
            source = os.path.join(input_images_path, filename)
            destination = os.path.join(output_images_path, filename)
            shutil.copy(source, destination)
            file_names.append(filename)

    # Function to get image annotations
    def get_img_ann(image_id):
        return [ann for ann in data['annotations'] if ann['image_id'] == image_id]

    # Function to get image data
    def get_img(filename):
        return next((img for img in data['images'] if img['file_name'] == filename), None)

    # Iterate through filenames and process each image
    for filename in file_names:
        img = get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']
        img_ann = get_img_ann(img_id)

        # Write normalized polygon data to a text file
        if img_ann:
            with open(os.path.join(output_labels_path, f"{os.path.splitext(filename)[0]}.txt"), "a") as file_object:
                for ann in img_ann:
                    current_category = ann['category_id'] - 1
                    polygon = ann['segmentation'][0]
                    normalized_polygon = [format(coord / img_w if i % 2 == 0 else coord / img_h, '.6f') for i, coord in enumerate(polygon)]
                    file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")

# Function to create a YAML file for the dataset
def create_yaml(input_json_path, output_yaml_path, train_path, val_path, test_path=None):
    with open(input_json_path) as f:
        data = json.load(f)
    
    # Extract the category names
    names = [category['name'] for category in data['categories']]
    
    # Number of classes
    nc = len(names)

    # Create a dictionary with the required content
    yaml_data = {
        'names': names,
        'nc': nc,
        'test': test_path if test_path else '',
        'train': train_path,
        'val': val_path
    }

    # Write the dictionary to a YAML file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)


if __name__ == "__main__":
    base_input_path = "/home/ubuntu/workspace/ml-datasets/dataset/"
    base_output_path = "/home/ubuntu/workspace/ml-datasets/yolo_dataset/"
    #/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/images/train
    #/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/annotations/val.json

    # Processing validation dataset (if needed)
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "yolo-data/images/val"),
        input_json_path=os.path.join(base_input_path, "yolo-data/annotations/val.json"),
        output_images_path=os.path.join(base_output_path, "valid/images"),
        output_labels_path=os.path.join(base_output_path, "valid/labels")
    )

    # Processing training dataset 
    convert_to_yolo(
        input_images_path=os.path.join(base_input_path, "yolo-data/images/train"),
        input_json_path=os.path.join(base_input_path, "yolo-data/annotations/train.json"),
        output_images_path=os.path.join(base_output_path, "train/images"),
        output_labels_path=os.path.join(base_output_path, "train/labels")
    )
    
    # Creating the YAML configuration file
    create_yaml(
        input_json_path=os.path.join(base_input_path, "yolo-data/annotations/train.json"),
        output_yaml_path=os.path.join(base_output_path, "data.yaml"),
        train_path="/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/images/train",
        val_path="/home/ubuntu/workspace/ml-datasets/dataset/yolo-data/images/val",
        test_path= None#'../test/images'  # or None if not applicable
    )
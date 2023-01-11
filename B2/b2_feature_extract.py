import os
import numpy as np
from keras.preprocessing import image

# PATH TO ALL IMAGES
global basedir

basedir = './Datasets'

def eye_col(line):
    """
    This function extracts the eye shape of an image

    Args:
        line: Line of the label.csv file representing an image

    Return:
        label of 0 (brown), 1 (blue), 2 (green), 3 (grey) or 4 (black) depending on the eye colour
    """
    split = line.split('\t')
    if split[1] == '0':
        return 0
    elif split[2] == '1':
        return 1
    elif split[2] == '2':
        return 2
    elif split[2] == '3':
        return 3
    else:
        return 4


def filename(line):
    """
    This function extracts the file name of an image

    Args:
        line: Line of the label.csv file representing an image

    Return:
        filename: Filename of the image
    """
    split = line.strip('\n').split('\t')
    filename = split[3]
    return filename


def extract_features_labels(data_filepath, labels_filepath):
    """
    This funtion extracts a colour image for all images in a given dataset folder,
    and the eye colour label for each image detected.

    Args: 
        data_filepath: String of the imag file path in the Datasets folder
        labels_filepath: String of the labels file path in the Datasets folder

    Return:
        features: A numpy array containing black and with images of size 500x500x3
        eye_labels: A numpy array containing the eye colour label (0 (brown), 1 (blue), 2 (green), 3 (grey) or 4 (black)) 
                    for each image in which a face was detected
    """
    images_dir = os.path.join(basedir, data_filepath)
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filepath), 'r')
    lines = labels_file.readlines()
    eye_col_labels = {filename(line) : eye_col(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('\\')[-1]
            # load colour image
            img = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic')).astype('uint8')
            all_features.append(img)
            all_labels.append(eye_col_labels[file_name])
            print(file_name, eye_col_labels[file_name])            

    features = np.array(all_features)
    eye_labels = np.array(all_labels) 
    print(features)
    return features, eye_labels
import os
import numpy as np
from keras.preprocessing import image

# PATH TO ALL IMAGES
global basedir
basedir = './Datasets'

def face_shape(line):
    """
    This function extracts the face shape of an image

    Args:
        line: Line of the label.csv file representing an image

    Return:
        label of 0, 1, 2, 3 or 4 depending on the face shape
    """
    split = line.split('\t')
    if split[2] == '0':
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
    This funtion extracts a black and white image for all images in a given dataset folder,
    and the  face shape label for each image detected.

    Args: 
        data_filepath: String of the img file path in the Datasets folder
        labels_filepath: String of the labels file path in the Datasets folder

    Return:
        features: A numpy array containing black and with images of size 500x500x3
        face_labels: A numpy array containing the face label (0, 1, 2 3 or 4) for each image in
                     which a face was detected
    """
    images_dir = os.path.join(basedir, data_filepath)
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filepath), 'r')
    lines = labels_file.readlines()
    face_shape_labels = {filename(line) : face_shape(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths:
            file_name= img_path.split('\\')[-1]
            # Load image in grey scale to reduce run time and image dimensions
            img = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               color_mode='grayscale',
                               interpolation='bicubic')).astype('uint8')
            all_features.append(img)
            all_labels.append(face_shape_labels[file_name])
            print(file_name, face_shape_labels[file_name])            

    features = np.array(all_features)
    face_labels = np.array(all_labels) 
    print(features)
    return features, face_labels
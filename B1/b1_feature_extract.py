import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import json

# PATH TO ALL IMAGES

global basedir

basedir = './Datasets'

# function to get the face shape from an image in cartoon dataset
def face_shape(line):
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

# function to get the filename of an image in cartoon dataset
def filename(line):
    split = line.strip('\n').split('\t')
    filename = split[3]
    return filename

def extract_features_labels(data_filepath, labels_filepath):
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
            # load image
            img = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic')).astype('uint8')

            all_features.append(img)
            all_labels.append(face_shape_labels[file_name])
            print(file_name, face_shape_labels[file_name])            

    features = np.array(all_features)
    face_labels = np.array(all_labels) 
    print(features)
    return features, face_labels
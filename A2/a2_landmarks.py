import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import json

# PATH TO ALL IMAGES

global basedir, image_paths, target_size

basedir = './Datasets'
detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(os.sys.path[0], 'A1\shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)


# how to find frontal human faces in an image using 68 landmarks.  These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth.

# The face detector we use is made using the classic Histogram of Oriented
# Gradients (HOG) feature combined with a linear classifier, an image pyramid,
# and sliding window detection scheme.  The pose estimator was created by
# using dlib's implementation of the paper:
# One Millisecond Face Alignment with an Ensemble of Regression Trees by
# Vahid Kazemi and Josephine Sullivan, CVPR 2014
# and was trained on the iBUG 300-W face landmark dataset (see https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#     C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#     300 faces In-the-wild challenge: Database and results.
#     Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def smile(line):
    """
    This function extracts the smile value of an image where -1 is not smiling and 1 is smiling

    Args:
        line: Line of the label.csv file representing an image

    Return:
        1 if the image is smiling and -1 if it is not smiling male
    """
    split = line.strip('\n').split('\t')
    if split[3] == '-1':
        return -1
    return 1

# function to get an images filename
def filename(line):
    """
    This function extracts the file name of an image

    Args:
        line: Line of the label.csv file representing an image

    Return:
        filename: Filename of the image
    """
    split = line.split('\t')
    filename = split[1]
    return filename

def extract_features_labels(data_filepath, labels_filepath, is_test):
    """
    This funtion extracts the landmarks features for all images in a specified dataset folder.
    It also extracts the smile label for each image.
    
    Args: 
        data_filepath: String of the img file path in the Datasets folder
        labels_filepath: String of the labels file path in the Datasets folder
        is_test: 0 or 1 value denoting the data as the training or test set

    Return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        smile_labels:      an array containing the smile label (not smiling=0 and smiling=1) for each image in
                            which a face was detected
    """
    i = 0
    images_dir = os.path.join(basedir, data_filepath)
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    target_size = None
    labels_file = open(os.path.join(basedir, labels_filepath), 'r')
    lines = labels_file.readlines()
    smile_labels = {filename(line) : smile(line) for line in lines[1:]}
    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        
        for img_path in image_paths:
            file_name= img_path.split('\\')[-1]
            # load image
            img = image.image_utils.img_to_array(
                image.image_utils.load_img(img_path,
                               target_size=target_size,
                               interpolation='bicubic'))
            features, _ = run_dlib_shape(img)
            if features is not None:
                all_features.append(features)
                all_labels.append(smile_labels[file_name])
                print(file_name, smile_labels[file_name])
            #     #LIMITS TO 50 IMAGES FOR TESTING AND SPEED
            #     i += 1
            # if i == 50:
            #     print('USE 50 IMAGES TO RUN FASTER')
            #     break
        all_features = [feature.tolist() for feature in all_features]
        all_labels = [(label + 1)/2 for label in all_labels] # simply converts the -1 into 0, so not smiling=0 and smiling=1

        data = {'features': all_features, 'labels' : all_labels}

        if is_test == True:
            data_filename = 'A2/test_data.json'
        else:
            data_filename = 'A2/training_data.json'

    output_file = open(data_filename, 'w')
    json.dump(data, output_file, indent = 3)
    output_file.close()


    landmark_features = np.array(all_features)
    smile_labels = (all_labels) 
    #print(landmark_features)
    return landmark_features, smile_labels
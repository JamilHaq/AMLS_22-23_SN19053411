from . import b1_feature_extract as b1_extract
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn




def classification():
    train_X, train_Y = b1_extract.extract_features_labels('cartoon_set\img', 'cartoon_set\labels.csv') 
    test_X, test_Y = b1_extract.extract_features_labels('cartoon_set_test\img', 'cartoon_set_test\labels.csv') 
    # print(len(train_Y), len(train_X))
    # print(train_X.shape)
    # print(train_X[0])
    train_X = train_X.reshape(train_X.shape[0], 500*500*3)
    train_Y = train_Y.astype(int)
    test_X = test_X.reshape(test_X.shape[0], 500*500*3)
    test_X = test_X.astype(int)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(750000,), activation='softmax',))
    #model = keras.Sequential([keras.layers.Dense(5, input_shape=(750000,), activation='sigmoid')])

    classifier = model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    histo = model.fit(train_X, train_Y, epochs=10, validation_data=(valid_X, valid_Y))
    predicted = model.predict(test_X)
    predicted_labels = [np.argmax(i) for i in predicted]
    test_error, test_accuracy = model.evaluate(test_X, test_Y, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    confusion_m = tf.math.confusion_matrix(labels=test_Y, predictions=predicted_labels)
    print(predicted[0])
    print(confusion_m)
    #print("Classification Report: \n", classification_report(test_Y, predicted, zero_division=0))
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_m, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()




# img_width, img_height = 224, 224 # Default input size for VGG16
# train_path = './Datasets\cartoon_set'
# test_path = './Datasets\cartoon_set_test'
# vgg = VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height, 3))

# # Show architecture
# #vgg.summary()

# # Extract features
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 32

# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
#     labels = np.zeros(shape=(sample_count, 5))
#     # Preprocess data
#     generator = datagen.flow_from_directory(directory,
#                                             target_size=(img_width,img_height),
#                                             batch_size = batch_size,
#                                             class_mode='categorical')
#     # Pass data through convolutional base
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = vgg.predict(inputs_batch)
#         features[i * batch_size: (i + 1) * batch_size] = features_batch
#         labels[i * batch_size: (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels
    
# train_features, train_labels = extract_features(train_path, 10000) 
# validation_features, validation_labels = extract_features(test_path, 2500)
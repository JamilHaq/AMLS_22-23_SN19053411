from . import b2_feature_extract as b2_extract
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sn

def accuracy_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Validation'])
    plt.show()

def loss_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def classification():
    train_X, train_Y = b2_extract.extract_features_labels('cartoon_set\img', 'cartoon_set\labels.csv') 
    test_X, test_Y = b2_extract.extract_features_labels('cartoon_set_test\img', 'cartoon_set_test\labels.csv') 
    print(len(train_Y), len(train_X))
    print(train_X.shape)
    print(train_X[0])
    # Used to check the image dimentions and what it looks like
    # print(train_X[0].shape)
    # plt.matshow(train_X[0])
    # plt.show()
    train_X = train_X.reshape(train_X.shape[0], 500*500*3)
    train_Y = train_Y.astype(int)
    test_X = test_X.reshape(test_X.shape[0], 500*500*3)
    test_Y = test_Y.astype(int)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=1)

    model = keras.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(750000,), activation='softmax',))
    #model.add(keras.layers.Dense(5, activation='softmax'))

    classifier = model.compile(optimizer=Adam(lr=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    history = model.fit(train_X, train_Y, epochs=15, validation_data=(valid_X, valid_Y))
    accuracy_graph(history)
    loss_graph(history)
    predicted = model.predict(test_X)
    predicted_labels = [np.argmax(i) for i in predicted]
    test_error, test_accuracy = model.evaluate(test_X, test_Y, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    confusion_m = tf.math.confusion_matrix(labels=test_Y, predictions=predicted_labels)
    print(predicted[0])
    print(confusion_m)
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_m, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
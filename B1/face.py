from . import b1_feature_extract as b1_extract
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sn

#Function to plot a graph of training vs test accuracy over multiple epochs
def accuracy_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training','Validation'])
    plt.show()

#Function to plot a graph of training vs test loss over multiple epochs
def loss_graph(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

#Simple neural network function to classify face shape from the cartoon datasets 
def classification():
    #Extract and reshape test and training data for use with the neural network
    train_X, train_Y = b1_extract.extract_features_labels('cartoon_set\img', 'cartoon_set\labels.csv') 
    test_X, test_Y = b1_extract.extract_features_labels('cartoon_set_test\img', 'cartoon_set_test\labels.csv') 
    print(len(train_Y), len(train_X))
    print(train_X.shape)
    print(train_X[0])
    # Used to check the image dimentions and what it looks like
    # print(train_X[0].shape)
    # plt.matshow(train_X[0])
    # plt.show()
    train_X = train_X.reshape(train_X.shape[0], 500*500)
    train_Y = train_Y.astype(int)
    test_X = test_X.reshape(test_X.shape[0], 500*500)
    test_Y = test_Y.astype(int)

    #Split training and test/validation data into random subsets
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=1)

    #Create neural network model and add input and output layer with appropriate neurons
    model = keras.Sequential()
    model.add(keras.layers.Dense(5, input_shape=(250000,), activation='softmax',))

    #Compile model by selecting ompimizer, learing rate, loss function and success metric
    classifier = model.compile(optimizer=Adam(lr=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    #Set how many epochs to train model for, pass in relevant datasets for training or validation
    history = model.fit(train_X, train_Y, epochs=20, validation_data=(valid_X, valid_Y))

    predicted = model.predict(test_X)
    predicted_labels = [np.argmax(i) for i in predicted]
    test_error, test_accuracy = model.evaluate(test_X, test_Y, verbose=1)
    print('Test error: {}, Test accuracy: {}'.format(test_error, test_accuracy))
    #Confusion matrix showing correcly predicted face shapes on the test set (ideal is 500 in the diagonal line)
    confusion_m = tf.math.confusion_matrix(labels=test_Y, predictions=predicted_labels)
    print(confusion_m)
    accuracy_graph(history)
    loss_graph(history)
    plt.figure(figsize = (10,7))
    sn.heatmap(confusion_m, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()



#TESTING OTHER MODELS REMOVE BEFORE SUBMITTING  
# def b1_get_data():
#     train_X, train_Y = b1_extract.extract_features_labels('cartoon_set\img', 'cartoon_set\labels.csv') 
#     test_X, test_Y = b1_extract.extract_features_labels('cartoon_set_test\img', 'cartoon_set_test\labels.csv')
#     # plt.imshow(train_X[0])
#     # plt.show()
#     train_X = train_X.reshape(train_X.shape[0], 500*500) #Flatten the training images
#     #train_X = train_X/225 #Normalise data
#     test_X = test_X.reshape(test_X.shape[0], 500*500) #Flatten the test images
#     #test_X = test_X/225 #Normalise data
#     test_Y = test_Y.astype(int)
#     train_Y = train_Y.astype(int)
#     tr_X = train_X
#     tr_Y = train_Y
#     te_X = test_X
#     te_Y = test_Y
#     return tr_X, tr_Y, te_X, te_Y

# #Function to one hot encode the lable data
# def one_hot(y, classes):
#     # A zero matrix of size (m, c)
#     y_hot = np.zeros((len(y), classes))
#     # Putting 1 for column where the given label is, using multidimensional indexing.
#     y_hot[np.arange(len(y)), y] = 1
#     return y_hot

# #Softmax calculating function
# def softmax(z):
#     # z--> linear part.
#     # subtracting the max of z for numerical stability.
#     exp = np.exp(z - np.max(z))
#     # Calculating softmax for all examples.
#     for i in range(len(z)):
#         exp[i] /= np.sum(exp[i])  
#     return exp

# #Softmax regression training function 
# def fit(X, Y, learing_rate, classes, epochs):    
#     m, n = X.shape  # n = number of features, m = number of training examples
#     # Randomly initializing weights and bias
#     w = np.random.random((n, classes))
#     b = np.random.random(classes)
#     # Empty list to store losses.
#     losses = []
    
#     # Training loop.
#     for epoch in range(epochs):
#         # Calculating hypothesis/prediction.
#         z = X@w + b
#         print(z)
#         y_hat = softmax(z)
#         y_hot = one_hot(Y, classes)# One-hot encoding y
#         # Calculating the gradient of loss w.r.t w and b.
#         w_grad = (1/m)*np.dot(X.T, (y_hat - y_hot)) 
#         b_grad = (1/m)*np.sum(y_hat - y_hot)       
#         # Updating the parameters.
#         w = w - learing_rate*w_grad
#         b = b - learing_rate*b_grad 
#         # Calculating loss and appending it in the list.
#         loss = -np.mean(np.log(y_hat[np.arange(len(Y)), Y]))
#         losses.append(loss)
#         # Printing out the loss at every 100th iteration.
#         if epoch%100==0:
#             print('Epoch {epoch}==> Loss = {loss}'
#                   .format(epoch=epoch, loss=loss))
#     return w, b, losses


# def predict(X, w, b):
#     # Predicting
#     z = X@w + b
#     y_hat = softmax(z)
#     #Returns the class with highest probability.
#     return np.argmax(y_hat, axis=1)

# def accuracy(y, y_hat):
#     return np.sum(y==y_hat)/len(y)
from . import landmarks as lmarks
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
import os
import json

def test():
    #Extracts training features from data
    train_X, train_y = lmarks.extract_features_labels('celeba\img', 'celeba\labels.csv') 
    #Extracts test features from data
    test_X, test_y = lmarks.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv') 
    male = [train_X[i] for i in range(len(train_X)) if train_y[i] == 0]
    female = [train_X[i] for i in range(len(train_X)) if train_y[i] == 1]
    x_male = [male[i][0] for i in range (len(male))]
    x_male2 = [x_male[i][0] for i in range (len(x_male))]
    y_male = [male[i][1] for i in range (len(male))]
    y_male2 = [y_male[i][1] for i in range (len(y_male))]
    print(x_male)
    print(y_male)
    print(x_male2)
    print(y_male2)
    plt.scatter(x_male, y_male)
    plt.show()
    return

#Extracts training and test features from data or file if they exist
def get_data():
    if not os.path.exists('A1/training_data.json'):
        train_X, train_Y = lmarks.extract_features_labels('celeba\img', 'celeba\labels.csv', is_test = False)
        #train_Y = np.array([y, -(y - 1)]).T  
        test_X, test_Y = lmarks.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv', is_test = True) 
        #test_Y = np.array([test_y, -(test_y - 1)]).T

    else:
        train = open('A1/training_data.json')
        training_data = json.load(train)
        train_X = np.array(training_data['features'])
        train_X = train_X.reshape(train_X.shape[0], -1)
        train_Y = np.array(training_data['labels'])
        #train_Y = np.array([y, -(y - 1)]).T 
        test = open('A1/test_data.json')
        testing_data = json.load(test)
        test_X = np.array(testing_data['features'])
        test_X = test_X.reshape(test_X.shape[0], -1)
        test_Y = np.array(testing_data['labels'])
        #test_Y = np.array([test_y, -(test_y - 1)]).T

    tr_X = train_X
    tr_Y = train_Y
    te_X = test_X
    te_Y = test_Y
    return tr_X, tr_Y, te_X, te_Y

#Function to try different C and gamma hyperparameters for SVMs
def SVM_selection(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC()
    param_grid = {'C': [0.1, 1], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear']}
    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    grid.fit(training_images, training_labels)
    print(grid.best_params_)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(test_images)
    print(classification_report(test_labels, grid_predictions))
    return 


#Uses SVM classification to train the model and then shows the accuracy
def img_SVM(training_images, training_labels, test_images, test_labels):
    C = 1.0  # SVM regularization parameter
    gamma = 0.0001 #Kernel coefficient primarily for ‘rbf’, ‘poly’
    classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)  #by default the kernel is RBF, kernel='linear', kernel='poly' ,degree=3
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    #print(pred)
    print("Accuracy:", accuracy_score(test_labels, pred))
    return pred 

def gender_detect_test():
    tr_X, tr_Y, te_X, te_Y = get_data()
    pred = img_SVM(tr_X, tr_Y, te_X, te_Y)
    print(pred)

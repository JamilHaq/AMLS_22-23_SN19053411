from . import a2_landmarks as lmarks2
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt

def a2_test():
    #Extracts training features from data
    train_X, train_y = lmarks2.extract_features_labels('celeba\img', 'celeba\labels.csv') 
    #Extracts test features from data
    test_X, test_y = lmarks2.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv') 
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

def a2_get_data():
    #Extracts training features from data
    train_X, y = lmarks2.extract_features_labels('celeba\img', 'celeba\labels.csv') 
    train_Y = np.array([y, -(y - 1)]).T
    #Extracts test features from data
    test_X, test_y = lmarks2.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv') 
    test_Y = np.array([test_y, -(test_y - 1)]).T
    tr_X = train_X
    tr_Y = train_Y
    te_X = test_X
    te_Y = test_Y
    return tr_X, tr_Y, te_X, te_Y

def a2_img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.LinearSVC()  #by default the kernel is RBF, kernel='linear', kernel='poly' ,degree=3
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    #print(pred)
    print("Accuracy:", accuracy_score(test_labels, pred))
    return pred 

def smile_detect_test():
    tr_X, tr_Y, te_X, te_Y = a2_get_data()
    pred = a2_img_SVM(tr_X.reshape((tr_X.shape[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((te_X.shape[0], 68*2)), list(zip(*te_Y))[0])
    print(pred)
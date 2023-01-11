from . import landmarks as lmarks
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm

def get_data():
    """
    This function obtains the test and training data from the celeba datasets

    Return:
        tr_X, tr_Y: Numpy array of training data landmark points, numpy array of training data gender labels
        te_X, te_Y: Numpy array of test data landmark point, numpy array of test data gender labels
    """
    train_X, train_Y = lmarks.extract_features_labels('celeba\img', 'celeba\labels.csv', is_test = False)
    test_X, test_Y = lmarks.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv', is_test = True) 

    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    tr_X = train_X
    tr_Y = train_Y
    te_X = test_X
    te_Y = test_Y
    return tr_X, tr_Y, te_X, te_Y


def SVM_selection(training_images, training_labels, test_images, test_labels):
    """
    This function tests different hyperparameters (C, gamma and degree) and kernels for SVMs on the training dataset. 
    It selects the best paramaters and prints a report of these on the test dataset

    Args:
        training_images: Numpy array of training data landmark points 
        training_labels: Numpy array of training data gender labels
        test_images: Numpy array of test data landmark point
        test_labels: Numpy array of test data gender labels
    """
    classifier = svm.SVC()
    #Parameter grid to test across, used degree for poly and gamma for rbf
    param_grid = {'C': [0.1, 1], 
              #'degree': [3, 4, 5],
              'gamma': [0.1, 0.01, 0.001, 0.0001],
              'kernel': ['poly, rbf, linear']}
    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)
    grid.fit(training_images, training_labels)
    print(grid.best_params_)
    print(grid.best_estimator_)
    grid_predictions = grid.predict(test_images)
    print(classification_report(test_labels, grid_predictions))
    return 

def img_SVM(training_images, training_labels, test_images, test_labels):
    """
    This function uses SVM classification with a polynomial kernel to train the model on the training data.
    C = 0.1, degree = 4
    It then predicts the gender labels of the test data and prints an calssification report

    Args:
        training_images: Numpy array of training data landmark points 
        training_labels: Numpy array of training data gender labels
        test_images: Numpy array of test data landmark point
        test_labels: Numpy array of test data gender labels

    Return:
        pred: Numpy array of predicted gender labels on the test data
    """
    C = 0.1  #SVM regularization parameter
    deg = 4 #Degree of kernel function, used only for rbf and poly
    gamma = 0.0001 #Kernel coefficient, used only for rbf
    classifier = svm.SVC(kernel='poly', C=C, degree=deg)
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    #print(pred)
    print("Classification Report: \n", classification_report(test_labels, pred, zero_division=0))
    print("Accuracy:", accuracy_score(test_labels, pred))
    return pred 

def gender_detect_test():
    """
    This function is the final task A1 function to get the test and training data and feed it into the best SVM.
    It prints a classification report for the gender predictions on the test data.
    """
    tr_X, tr_Y, te_X, te_Y = get_data()
    pred = img_SVM(tr_X, tr_Y, te_X, te_Y)
    print(pred)

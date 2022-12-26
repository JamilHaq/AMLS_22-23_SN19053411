from . import landmarks as lmarks
from sklearn import svm

def test():
    Xtest, Ytest = lmarks.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv')
    print(Xtest[:100])
    print(Ytest[:100])
    return

def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    #print(pred)
    print("Accuracy:", lmarks.accuracy_score(test_labels, pred))
    return pred 
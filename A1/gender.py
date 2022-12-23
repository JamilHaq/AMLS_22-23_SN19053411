from . import landmarks as lmarks

def test():
    Xtest, Ytest = lmarks.extract_features_labels('celeba_test\img', 'celeba_test\labels.csv')
    print(Xtest[:100])
    print(Ytest[:100])
    return
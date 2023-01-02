from A1 import gender as a1
from A2 import emotion as a2

#a1.test()
#a1.gender_detect_test()

#tr_X, tr_Y, te_X, te_Y = a1.get_data()
#a1.img_SVM(tr_X, tr_Y, te_X, te_Y)
#a1.gender_detect_test()
#a1.SVM_selection(tr_X, tr_Y, te_X, te_Y)


a2.smile_detect_test()
# tr_X, tr_Y, te_X, te_Y = a2.a2_get_data()
# a2.a2_SVM_selection(tr_X, tr_Y, te_X, te_Y)

#For gender detection (with C and gamma unchanged):
# SVC with Gaussian radial basis function, RBF, (degree 3 by default) kernel has an accuracy of about 85.57%
# SVC with linear kernel has an accuracy of about 92.47%
# SVC with .LinearSVC (linear kernel) has an accuracy of about 91.23%
# SVC with polynomial (degree 3) kernel has an accuracy of about 92.27%
# SVC(C=1, gamma=1, kernel='linear')
#               precision    recall  f1-score   support

#          0.0       0.91      0.95      0.93       489
#          1.0       0.95      0.90      0.92       481

#     accuracy                           0.92       970
#    macro avg       0.93      0.92      0.92       970
# weighted avg       0.93      0.92      0.92       970

# {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
# SVC(C=1, gamma=0.0001)
#               precision    recall  f1-score   support

#          0.0       0.89      0.94      0.91       489
#          1.0       0.93      0.88      0.91       481

#     accuracy                           0.91       970
#    macro avg       0.91      0.91      0.91       970
# weighted avg       0.91      0.91      0.91       970



#For smile detection:
# SVC with Gaussian radial basis function, RBF, (degree 3 by default) kernel has an accuracy of about 88.76%
# SVC with linear kernel has an accuracy of about 90%
# SVC with .LinearSVC (linear kernel) has an accuracy of about 52.89%
# SVC with polynomial (degree 3) kernel has an accuracy of about 90.52%
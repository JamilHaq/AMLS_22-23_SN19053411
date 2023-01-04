from A1 import gender as a1
from A2 import emotion as a2

#a1.test()
#a1.gender_detect_test()

#tr_X, tr_Y, te_X, te_Y = a1.get_data()
#a1.img_SVM(tr_X, tr_Y, te_X, te_Y)
a1.gender_detect_test()
#a1.SVM_selection(tr_X, tr_Y, te_X, te_Y)


#a2.smile_detect_test()
#tr_X, tr_Y, te_X, te_Y = a2.a2_get_data()
#a2.a2_SVM_selection(tr_X, tr_Y, te_X, te_Y)


########################################################################
# A1
#For gender detection (with best C, gamma and degree values):
# SVC with radial basis function, RBF, kernel has an accuracy of about 91.03% (C=1, gamma=0.0001, degree=1)
# SVC with linear kernel has an accuracy of about 92.47% (C=1)
# SVC with .LinearSVC (linear kernel) has an accuracy of about 91.23% 
# SVC with polynomial kernel has an accuracy of about 92.47% (C=0.1, degree=4)

# {'C': 1, 'degree': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
# SVC(C=1, degree=1, gamma=0.0001)
#               precision    recall  f1-score   support

#          0.0       0.89      0.94      0.91       489
#          1.0       0.93      0.88      0.91       481

#     accuracy                           0.91       970
#    macro avg       0.91      0.91      0.91       970
# weighted avg       0.91      0.91      0.91       970

# SVC(C=1, kernel='linear')
#               precision    recall  f1-score   support

#          0.0       0.91      0.95      0.93       489
#          1.0       0.95      0.90      0.92       481

#     accuracy                           0.92       970
#    macro avg       0.93      0.92      0.92       970
# weighted avg       0.93      0.92      0.92       970

# SVC(C=0.1, degree=4, kernel='poly') accuracy = 92.47%
#               precision    recall  f1-score   support

#          0.0       0.91      0.94      0.93       489
#          1.0       0.94      0.91      0.92       481

#     accuracy                           0.92       970
#    macro avg       0.93      0.92      0.92       970
# weighted avg       0.93      0.92      0.92       970


###############################################################################
# A2
#For smile detection (with best C, gamma and degree values):
# SVC with radial basis function, RBF, kernel has an accuracy of about 88.76% (C=1, deg=2, gamma=0.0001) (89.93% with C=1 and gamma=0.0001)
# SVC with linear kernel has an accuracy of about 90% (C=0.1)
# SVC with .LinearSVC (linear kernel) has an accuracy of about 52.89%
# SVC with polynomial kernel has an accuracy of about 90.31% (C=0.1, degree=4) (90.52% with C=1 and deg=3 )

# {'C': 1, 'degree': 1, 'gamma': 0.0001, 'kernel': 'rbf'}
# SVC(C=1, degree=1, gamma=0.0001)
#               precision    recall  f1-score   support

#          0.0       0.89      0.89      0.89       473
#          1.0       0.90      0.89      0.90       497

#     accuracy                           0.89       970
#    macro avg       0.89      0.89      0.89       970
# weighted avg       0.89      0.89      0.89       970

# SVC(C=0.1, kernel='linear')
#               precision    recall  f1-score   support

#          0.0       0.89      0.90      0.90       473
#          1.0       0.91      0.90      0.90       497

#     accuracy                           0.90       970
#    macro avg       0.90      0.90      0.90       970
# weighted avg       0.90      0.90      0.90       970

# SVC(C=0.1, degree=4, kernel='poly')
#               precision    recall  f1-score   support

#          0.0       0.90      0.90      0.90       473
#          1.0       0.91      0.90      0.91       497

#     accuracy                           0.90       970
#    macro avg       0.90      0.90      0.90       970
# weighted avg       0.90      0.90      0.90       970
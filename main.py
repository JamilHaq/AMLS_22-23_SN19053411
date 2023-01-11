from A1 import gender as a1
from A2 import emotion as a2
from B1 import face as b1
from B2 import eye as b2

#a1.test()
#a1.gender_detect_test()

#tr_X, tr_Y, te_X, te_Y = a1.get_data()
#a1.img_SVM(tr_X, tr_Y, te_X, te_Y)
#a1.gender_detect_test()
#a1.SVM_selection(tr_X, tr_Y, te_X, te_Y)

#a2.smile_detect_test()
#tr_X, tr_Y, te_X, te_Y = a2.a2_get_data()
#a2.a2_SVM_selection(tr_X, tr_Y, te_X, te_Y)

# tr_X, tr_Y, te_X, te_Y = b1.b1_get_data()
# w, b, l = b1.fit(tr_X, tr_Y, learing_rate=0.9, classes=5, epochs=10)
# train_preds = b1.predict(tr_X, w, b)
# b1.accuracy(tr_Y, train_preds)
# test_preds = b1.predict(te_X, w, b)
# b1.accuracy(te_Y, test_preds)
b1.classification()

#b2.classification() 

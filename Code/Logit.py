import numpy as np
import pandas as pd
import os
import sklearn 
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

#--------data_file_address -------#
cwd = os.getcwd()
train_data_dir = os.pardir + "\data\small\SMALLER_TRAIN.csv"
test_data_dir = os.pardir + "\data\small\SMALLER_Test.csv"

#---------loading data---------#
data_train = np.genfromtxt(train_data_dir,delimiter=",")
data_test = np.genfromtxt(test_data_dir, delimiter = ",")

m,n = np.shape(data_train)

X_train = data_train[:,: 11]
y_train = data_train[:, n - 1]
X_test = data_test[:, : 11]
y_test = data_test[:, n - 1]

# #---------training model---------#
# logit_cl = linear_model.LogisticRegression()
# logit_cl.fit(X_train, y_train)
# y_predict = logit_cl.predict(X_test)
# c_m = confusion_matrix(y_test, y_predict)
# acc_rate = np.trace(c_m)/len(y_test)
#
# print(acc_rate)

#---------perform a 10-fold cross validation---------#
kf = KFold(n_splits= 10)
##---------initialize the result matrix---------##
result_matrix = np.ones((10,2))

##---------Trainning 10 models---------##
tf_index = list(kf.split(X_train, y_train))
logit_tf_cl =  linear_model.LogisticRegression()

for i in range(0,10):
	tf_X_train = data_train[tf_index[i][0]][:,: 11]
	tf_y_train = data_train[tf_index[i][0]][:, n - 1]

	tf_X_test = data_train[tf_index[i][1]][:,: 11]
	tf_y_test = data_train[tf_index[i][1]][:, n - 1]

	logit_tf_cl.fit(tf_X_train, tf_y_train)
	tf_y_predict = logit_tf_cl.predict(tf_X_test)

	tf_c_m = confusion_matrix(tf_y_test, tf_y_predict)
	tf_acc_rate = np.trace(tf_c_m)/(len(tf_X_train) + len(tf_X_test))

	print(tf_acc_rate)


	


	
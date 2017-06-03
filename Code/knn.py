import os
import numpy as np
import pandas as pd
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
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


#---------training model---------#
n_neighbors = 15
knn_cl = KNeighborsClassifier(n_neighbors)
knn_cl.fit(X_train, y_train)
y_predict = knn_cl.predict(X_test)

c_m = confusion_matrix(y_test, y_predict)
acc_rate = np.trace(c_m)/len(y_test)

print(acc_rate)

#
# #---------perform a 10-fold cross validation---------#
# kf = KFold(n_spplits= 10)
# ##---------initialize the result matrix---------##
# result_matrix = np.ones((10,2))
# ##---------experiment with different k---------##
# k_list = list(range(3,13))
# ##---------Trainning 10 models---------##
# for i in range(3, 13):
# 	tf_n_neighbors = i
# 	knn_tf_cl = KNeighborsClassifier(tf_n_neighbors)
# 	tf_train_data, tf_test_datatf_X_train = kf.split(train_data)[i]
#
# 	tf_m, tf_n = np.shape(train_data)
# 	tf_X_train = tf_train_data[:, (tf_n - 1)]
# 	tf_y_train = tf_train_data[:, tf_n]
#
# 	tf_X_test = tf_test_data[:, (tf_n - 1)]
# 	tf_y_test = tf_test_data[:, tf_n)]
#
#
# 	knn_tf_cl.fit(tf_X_train, tf_y_train)
#
# 	tf_y_predict = knn_tf_cl.predict(tf_y_test)
#
# 	tf_c_m = confusion_matrix(tf_y_test, tf_y_predict)
# 	tf_acc_rate = np.trace(tf_c_m)/len(y_test)
#
# 	print(tf_acc_rate)
#
#
	


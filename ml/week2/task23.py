from __future__ import print_function
import pandas

import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

data_train = pandas.read_csv('perceptron-train.csv')
data_test = pandas.read_csv('perceptron-test.csv')
total = len(data_train)


X_train = data_train[[1,2]]
Y_train = data_train.ix[:,0]

X_test = data_test[[1,2]]
Y_test = data_test.ix[:,0]
clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)
#print(predictions)
accuracy = sklearn.metrics.accuracy_score(Y_test,predictions)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, Y_train)
predictions = clf.predict(X_test_scaled)
#print(predictions)
accuracy_scaled = sklearn.metrics.accuracy_score(Y_test,predictions)

print(accuracy,accuracy_scaled,accuracy_scaled - accuracy)
#print(max_p, max_mean)
f1 = open('data/231.txt','w')
print(accuracy_scaled - accuracy, file=f1)


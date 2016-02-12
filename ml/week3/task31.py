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
from sklearn.svm import SVC

data_train = pandas.read_csv('svm-data.csv',header=None)
print(data_train)

X_train = data_train[[1,2]]
Y_train = data_train.ix[:,0]

clf = SVC(C=100000,kernel='linear',random_state=241)
clf.fit(X_train, Y_train)

print(clf.support_)

#predictions = clf.predict(X_test)
res = ''
for s in clf.support_:
	res += str(s+1) + ' '
print(res)

f1 = open('data/311.txt','w')
print(res, file=f1)


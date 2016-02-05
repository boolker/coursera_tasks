from __future__ import print_function
import pandas

import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

data = pandas.read_csv('wine.data')
total = len(data)
#print()
Y = data.ix[:,0]


X = data[[1,2,3,4,5,6,7,8,9,10,11,12,13]]
#print(Y,X)

folds = sklearn.cross_validation.KFold(total, n_folds=5, shuffle=True,random_state=42)
#for train, test in folds:
#	print(train, test)
max_k = -1
max_mean = 0
for k in xrange(1,51):
	kn = KNeighborsClassifier(n_neighbors=k)
	res = sklearn.cross_validation.cross_val_score(kn,X,Y.values,cv=folds)
	r_mean = res.mean()
	if r_mean > max_mean:
		max_mean = r_mean
		max_k = k
	#print(k, res, res.mean())
max_k_scaled = -1
max_mean_scaled = 0

X_scaled = preprocessing.scale(X)
for k in xrange(1,51):
	kn = KNeighborsClassifier(n_neighbors=k)
	res = sklearn.cross_validation.cross_val_score(kn,X_scaled,Y.values,cv=folds)
	r_mean = res.mean()
	if r_mean > max_mean_scaled:
		max_mean_scaled = r_mean
		max_k_scaled = k

print(max_k, max_mean,max_k_scaled,max_mean_scaled)
f1 = open('data/211.txt','w')
print(max_k, file=f1)
f2 = open('data/212.txt','w')
print(max_mean, file=f2)
f3 = open('data/213.txt','w')
print(max_k_scaled, file=f3)
f4 = open('data/214.txt','w')
print(max_mean_scaled, file=f4)

#########


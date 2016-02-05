from __future__ import print_function
import pandas

import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

boston = sklearn.datasets.load_boston()

data = boston['data']
target = boston['target']
total = len(data)

data_scaled = preprocessing.scale(data)
#print()
#print(data_scaled)
folds = sklearn.cross_validation.KFold(total, n_folds=5, shuffle=True,random_state=42)

max_p = -1
max_mean = -1000
for _p in np.linspace(1, 10, num=200):
	knn = KNeighborsRegressor(n_neighbors=5, weights='distance',p=_p)
	res = sklearn.cross_validation.cross_val_score(knn,data,target,cv=folds,scoring='mean_squared_error')
	r_mean = res.mean()
	print(res,_p,r_mean)	
	if r_mean > max_mean:
		max_mean = r_mean
		max_p = _p

print(max_p, max_mean)
f1 = open('data/221.txt','w')
print(max_p, file=f1)


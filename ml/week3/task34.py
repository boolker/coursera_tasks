from __future__ import print_function
import pandas
import math
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV

from sklearn import datasets
from sklearn.metrics import roc_auc_score

def calc_delta(_w,_w_old,_n):
	_sum = 0
	for i in xrange(_n):
		_sum += (_w[i]-_w_old[i])*(_w[i]-_w_old[i])
	return math.sqrt(_sum)

def regression(_X,_Y,_k,_C,_eps):
	w_old = [0,0]
	_w = [0,0]

	total = len(_Y)	

	for _it in xrange(10000):
		for k in xrange(2):
			_w[k] = w_old[k] - _k*_C*w_old[k]
			_sum = 0
			for i, x in _X.iterrows():
				_M = 0
				for j in xrange(2):
					_M += w_old[j]*x[j+1]
				_sum += _Y[i]*x[k+1]*(1 - 1.0/(1+math.exp(-1*_Y[i]*_M)))
			_w[k] = _w[k] + _sum*_k/total
		#	print(x[1],x[2])
		delta = calc_delta(_w,w_old,2)
		if delta < _eps:
			break
		for k in xrange(2):
			w_old[k] = _w[k]
		
	print(_w,_it)
	return _w

data_train = pandas.read_csv('data-logistic.csv',header=None)
print(data_train)

X = data_train[[1,2]]
Y = data_train.ix[:,0]

Y_bin = [0 if y == -1 else y for y in Y]
#print(Y)
#print(X,Y)
#for i, x in X.iterrows():
#	print(x[1],x[2])
total = len(Y)

w = regression(X,Y,0.1,10,1e-5)

Y_score = [0 for i in xrange(total)]
#print(Y_score)
for i, x in X.iterrows():
	_M = 0
	for j in xrange(2):
		_M += w[j]*x[j+1]
	#_M = _M*Y[i]
	Y_score[i] =  1.0/(1+math.exp(-1*_M))
#print(Y_score)
res = [0,0]
res[0] = roc_auc_score(Y, Y_score)

w = regression(X,Y,0.1,0,1e-5)

Y_score = [0 for i in xrange(total)]

for i, x in X.iterrows():
	_M = 0
	for j in xrange(2):
		_M += w[j]*x[j+1]
	#_M = _M*Y[i]
	Y_score[i] =  1.0/(1+math.exp(-1*_M))
#print(Y_score)
res[1] = roc_auc_score(Y, Y_score)

print(res)
res_str = ''
for r in res:
	res_str += '{0:.3f}'.format(r) + ' '
	#res_str += str(r) + ' '

print(res_str)
f1 = open('data/331.txt','w')
print(res_str, file=f1)


from __future__ import print_function
import pandas
import math
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn.svm import SVC

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


data_train = pandas.read_csv('gbm-data.csv')
#print(data_train)
Y = data_train.ix[:,0]

data_train = data_train.drop('Activity',1)
X = data_train.values
#print(X)

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X,Y,test_size = 0.8, random_state = 241)



#learning_rate = [1, 0.5, 0.3, 0.2, 0.1] 
learning_rate = [ 0.2] 
min_iter = -1
min_val = 100

plt.figure()

for rate in learning_rate:
	
	clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241,learning_rate = rate)
	clf.fit(X_train,Y_train)
	'''predict = clf.staged_decision_function(X_train)
	log_loss_train = []
	for p in predict:
		p_sig = [1.0/(1 + math.exp(-1*_p)) for _p in p]		
		log_loss_train.append(sklearn.metrics.log_loss(Y_train,p_sig))

	plt.plot(log_loss_train, 'r', linewidth=rate*10)'''

	predict_test = clf.staged_decision_function(X_test)
	log_loss_test = []
	for p in predict_test:
		p_sig = [1.0/(1 + math.exp(-1*_p)) for _p in p]		
		log_loss_test.append(sklearn.metrics.log_loss(Y_test,p_sig))
	plt.plot(log_loss_test, 'g', linewidth=rate*8)
	plt.legend(['test'])

	pred = clf.predict_proba(X_test)
	print(pred[:,1])

	if rate == 0.2:
		i = 0
		for l in log_loss_test:
			if l < min_val:
				min_val = l
				min_iter = i
			i += 1

plt.show()

f1 = open('data/521.txt','w')
print('overfitting', file=f1)

print(min_iter,min_val)
f2 = open('data/522.txt','w')
print(min_iter, '{0:.2f}'.format(min_val), file=f2)
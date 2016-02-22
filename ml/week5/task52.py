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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


data = pandas.read_csv('gbm-data.csv').values
#print(data_train)
Y = data[:,0]

#data_train = data_train.drop('Activity',1)
X = data[:,1:]
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
	
	predict_test = clf.staged_decision_function(X_test)
	log_loss_test = []
	for p in predict_test:
		#p_sig = [1.0/(1 + math.exp(-1*_p)) for _p in p]
		x = sklearn.metrics.log_loss(Y_test,1.0/(1.0+np.exp(-p)))		
		log_loss_test.append(x)
	#plt.plot(log_loss_test, 'g', linewidth=rate*8)
	#plt.legend(['test'])

	#pred = clf.predict_proba(X_test)
	#print(pred[:,1])
	print(log_loss_test)
	if rate == 0.2:
		i = 0
		for i, l in enumerate(log_loss_test):
			if l < min_val:
				min_val = l
				min_iter = i

#plt.show()

f1 = open('data/521.txt','w')
print('overfitting', file=f1)

print(min_val, min_iter)
f2 = open('data/522.txt','w')
print('{0:.2f}'.format(min_val),min_iter, file=f2)

f_clf = RandomForestClassifier(n_estimators=min_iter+1,random_state=241)
f_clf.fit(X_train,Y_train)
pred = f_clf.predict_proba(X_test)[:, 1]
print(pred)
x = sklearn.metrics.log_loss(Y_test,pred)
print(x)
f3 = open('data/523.txt','w')
print('{0:.2f}'.format(x), file=f3)

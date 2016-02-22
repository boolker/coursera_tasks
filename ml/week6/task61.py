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

from sklearn import metrics
from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

data_train = pandas.read_csv('abalone.csv')
#print(data_train)

data_train['Sex'] = data_train['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

Y = data_train['Rings']
data_train = data_train.drop('Rings',1)

#print(data_train)
#print(Y)
#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([-3, 1, 10])
_cv = sklearn.cross_validation.KFold(len(Y), n_folds=5, shuffle=True, random_state=1)

index = 1
for i in xrange(50):
	clf = RandomForestRegressor(n_estimators=i+1,random_state=1)
	
	res = sklearn.cross_validation.cross_val_score(clf,data_train,Y,cv=_cv,scoring='r2')
	print(res, res.mean())
	if res.mean() > 0.52:
		break
	index += 1
	#clf.fit(data_train, Y)
#predictions = clf.predict(X)
print(index)
#print(prices.columns.values[index+1])
f1 = open('data/511.txt','w')
print(index, file=f1)
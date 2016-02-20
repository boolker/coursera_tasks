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

data_train = pandas.read_csv('salary-train.csv')

data_test = pandas.read_csv('salary-test-mini.csv')

data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data_train['LocationNormalized'] = data_train['LocationNormalized'].str.lower()

#print(data_train)
vectorizer = TfidfVectorizer(min_df=5)
v = vectorizer.fit_transform(data_train['FullDescription'])
#print(v)
	
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

#print(X_train_categ)

uni = hstack([v,X_train_categ])
#print(uni.shape)

clf = Ridge(alpha=1.0)
y = data_train['SalaryNormalized']
clf.fit(uni, y)

v_test = vectorizer.transform(data_test['FullDescription'])

uni_test = hstack([v_test,X_test_categ])
print(uni_test.shape)

res = clf.predict(uni_test)
print(res)

res_str = ''
for r in res:
	res_str += '{0:.2f}'.format(r) + ' '

#print(max_score, column_name)
f1 = open('data/411.txt','w')
print(res_str, file=f1)





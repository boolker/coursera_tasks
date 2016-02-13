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


def get_p(_y_true,_y_pred,r_threshold):
	p = 0
	max_p = 0
	_p, _r, _t = metrics.precision_recall_curve(_y_true,_y_pred)
	#print(_p, _r)
	for i in xrange(len(_p)):
		if _r[i] >= r_threshold:
			if _p[i] > max_p:
				max_p = _p[i]
	return max_p


data_class = pandas.read_csv('classification.csv')
#print(data_class)


y_real = data_class.ix[:,0]
y_pred = data_class.ix[:,1]

tp = 0
fp = 0
tn = 0
fn = 0

for i in xrange(len(y_real)):
	if y_real[i] == 1:
		if y_pred[i] == 1:
			tp += 1
		else:
			fn += 1
	else:
		if y_pred[i] == 1:
			fp += 1
		else:
			tn += 1

acc = metrics.accuracy_score(y_real,y_pred)
precision = metrics.precision_score(y_real,y_pred)
recall = metrics.recall_score(y_real,y_pred)
f1_s = metrics.f1_score(y_real,y_pred)

print(tp,fp,fn,tn)
f1 = open('data/341.txt','w')
print(tp,fp,fn,tn, file=f1)

res_str = '{0:.2f}'.format(acc) + ' ' + '{0:.2f}'.format(precision) + ' ' + '{0:.2f}'.format(recall) + ' ' + '{0:.2f}'.format(f1_s)
f2 = open('data/342.txt','w')
print(res_str, file=f2)

scores = pandas.read_csv('scores.csv')
y_true = scores.ix[:,0]
y_log = scores.ix[:,1]
y_svm = scores.ix[:,2]
y_knn = scores.ix[:,3]
y_tree = scores.ix[:,4]
max_score = 0
column_name = ''
score = metrics.roc_auc_score(y_true,y_log)
print(score)
if score > max_score:
	max_score = score
	column_name = 'score_logreg'

score = metrics.roc_auc_score(y_true,y_svm)
print(score)
if score > max_score:
	max_score = score
	column_name = 'score_svm'

score = metrics.roc_auc_score(y_true,y_knn)
print(score)
if score > max_score:
	max_score = score
	column_name = 'score_knn'

score = metrics.roc_auc_score(y_true,y_tree)
print(score)
if score > max_score:
	max_score = score
	column_name = 'score_tree'

print(max_score, column_name)
f3 = open('data/343.txt','w')
print(column_name, file=f3)


log_p = get_p(y_true,y_log,0.7)
print(log_p)

svm_p = get_p(y_true,y_svm,0.7)
print(svm_p)

knn_p = get_p(y_true,y_knn,0.7)
print(knn_p)

tree_p = get_p(y_true,y_tree,0.7)
print(tree_p)

max_score = 0
column_name = ''
if log_p > max_score:
	max_score = log_p
	column_name = 'score_logreg'

if svm_p > max_score:
	max_score = svm_p
	column_name = 'score_svm'

if knn_p > max_score:
	max_score = knn_p
	column_name = 'score_knn'

if tree_p > max_score:
	max_score = tree_p
	column_name = 'score_tree'

print(max_score, column_name)
f4 = open('data/344.txt','w')
print(column_name, file=f4)





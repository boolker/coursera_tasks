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

djia = pandas.read_csv('djia_index.csv')

#djia_del = djia.drop('date',1)
#print(djia)
prices = pandas.read_csv('close_prices.csv')
prices_del = prices.drop('date',1)
pca = PCA(n_components=10)

pca.fit(prices_del)
print(pca.explained_variance_ratio_)
_sum = 0
count = 0
for r in pca.explained_variance_ratio_:
	_sum += r
	count +=1
	if _sum >= 0.9:
		break
print(count)
f1 = open('data/421.txt','w')
print(count, file=f1)


n_prices = pca.transform(prices_del)

print(len(n_prices))
comp_1 = n_prices[:,0]
d = djia.ix[:,1]
#print(n_prices[:,0])

corr = np.corrcoef(comp_1,d) 
print(corr[0][1])
f2 = open('data/422.txt','w')
print('{0:.2f}'.format(corr[0][1]), file=f2)

print(len(pca.components_))
print(pca.components_[0])

max_c = 0
index = -1
i = 0
for c in pca.components_[0]:
	print(i, c)
	if abs(c) > max_c:
		max_c = abs(c)
		index = i
	i += 1

print('res: ',index, max_c)	

print(prices.columns.values[index+1])
f3 = open('data/423.txt','w')
print(prices.columns.values[index+1], file=f3)

#res_str = ''
#for r in res:
#	res_str += '{0:.2f}'.format(r) + ' '

#print(max_score, column_name)
#f1 = open('data/421.txt','w')
#print(res_str, file=f1)





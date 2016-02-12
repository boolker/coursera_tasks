from __future__ import print_function
import pandas

import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn import datasets
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV

from sklearn import datasets

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

data = newsgroups.data
target = newsgroups.target

vectorizer = TfidfVectorizer(min_df=1)
v = vectorizer.fit_transform(data)
#print(v)

total = len(target)
grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = sklearn.cross_validation.KFold(total, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
res = gs.fit(v, target)

#print(v)
#print(target)
print(gs.best_params_)

new_clf = SVC(gs.best_params_['C'],kernel='linear',random_state=241)
new_clf.fit(v, target)

#print(len(new_clf.coef_.indices),len(new_clf.coef_.data),len(v))
#print(new_clf.coef_)
#print('test')
#print(new_clf.coef_[0])
words = np.argsort(np.absolute(np.asarray(new_clf.coef_.todense())).reshape(-1))[-10:]
names = vectorizer.get_feature_names()
res_names = []
res = ''
for w in words:
	res_names.append(names[w])
	print(names[w])

res_names.sort()
print(res_names)

for w in res_names:
	res += w + ' '

print(res)
f1 = open('data/321.txt','w')
print(res, file=f1)


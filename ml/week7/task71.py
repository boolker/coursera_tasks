from __future__ import print_function
import pandas
import math
import numpy as np
import sklearn
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import time
import datetime


def boosting_task(_data):
	#get count of rows in dataset
	total = data.shape[0]
	Y = data['radiant_win']
	X = data.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_dire','barracks_status_radiant'],1)

	column_list = []
	for i, x in enumerate(X.count()):
		if x < total:
			column_list.append(data.columns.values[i])
			#print(x, data.columns.values[i])

	X = X.fillna(0)

	trees = [10,20,30,40]
	learning_rate = [0.5, 0.3, 0.2]

	_cv = sklearn.cross_validation.KFold(total, n_folds=5, shuffle=True, random_state=1)

	#way 1
	performance_res = []
	for t in trees:
		for rate in learning_rate:
			clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=t, verbose=True, random_state=241,learning_rate = rate)
			start_time = datetime.datetime.now()
			res = sklearn.cross_validation.cross_val_score(clf,X,Y,cv=_cv,scoring='roc_auc')
			elapsed = datetime.datetime.now() - start_time
			performance_res.append([t, rate, res.mean(), elapsed])

	for p in performance_res:
		print(p[0], '\t\t\t\t', p[1], '\t\t',p[2],'\t',p[3])

	#way 2
	'''grid = {'n_estimators': trees}
	clf = GradientBoostingClassifier(verbose=True, random_state=241)
	gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=_cv)
	res = gs.fit(X, Y)

	print(gs.best_params_)
	print(gs.grid_scores_)'''


######################### LOGISTIC REGRESSION  #######################################	
def cal_log_reg_quality(_C,_X,_Y,_cv):
	LR_performance_res = []
	best_res = 0
	best_C = 0
	for c in _C:
		clf_lr = LogisticRegression(penalty='l2',C=c)
		start_time = datetime.datetime.now()
		res = sklearn.cross_validation.cross_val_score(clf_lr,_X,_Y,cv=_cv,scoring='roc_auc')
		elapsed = datetime.datetime.now() - start_time
		LR_performance_res.append([c, res.mean(), elapsed])
		if res.mean() > best_res:
			best_res = res.mean()
			best_C = c

	print(best_res,best_C)
	for p in LR_performance_res:
		print(p[0], '\t\t\t\t', p[1], '\t\t',p[2])

	return best_C

def regression_task(_data,_data_test):
	#C_vals = np.power(10.0, np.arange(-5, 5))
	C_vals = np.power(10.0, np.arange(-3, 2))

	total = _data.shape[0]
	Y = _data['radiant_win']
	X = _data.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_dire','barracks_status_radiant'],1)
	X = X.fillna(0)
	cv = sklearn.cross_validation.KFold(total, n_folds=5, shuffle=True, random_state=1)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	print('LR with categorial params')
	best_C = cal_log_reg_quality(C_vals,X_scaled,Y,cv)
	
	#check coeffs
	#clf_lr = LogisticRegression(penalty='l2',C=best_C)
	#clf_lr.fit(X_scaled,Y)
	#print(clf_lr.coef_)

	X_no_categ = X.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero',
		'd1_hero','d2_hero','d3_hero','d4_hero','d5_hero'],1)
	
	X_scaled = scaler.fit_transform(X_no_categ)
	
	print('LR without categorial params')
	best_C = cal_log_reg_quality(C_vals,X_scaled,Y,cv)

	#2.3 
	heroes = [X['r1_hero'],X['r2_hero'],X['r3_hero'],X['r4_hero'],X['r5_hero'],
				X['d1_hero'],X['d2_hero'],X['d3_hero'],X['d4_hero'],X['d5_hero']]
	all_heroes = pandas.concat(heroes)

	#get max hero id
	max_id = 0
	for _id in all_heroes.unique():
		if _id > max_id:
			max_id = _id
		
	N = len(all_heroes.unique())

	X_pick = np.zeros((X.shape[0], max_id))

	#build bag of params
	for i, match_id in enumerate(X.index):
	    for p in xrange(5):
	        X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
	        X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

	#concatenate dataframes
	X_concat = pandas.concat([X_no_categ,pandas.DataFrame(X_pick,index=X_no_categ.index)],axis=1)
	print(X_no_categ.shape, X_pick.shape, X_concat.shape)

	X_scaled = scaler.fit_transform(X_concat)
	print('LR with bag of params')
	best_C_bag = cal_log_reg_quality(C_vals,X_scaled,Y,cv)
	
	clf_best = LogisticRegression(penalty='l2',C=best_C_bag)
	clf_best.fit(X_scaled,Y)

	#preprocess test data
	X_test = _data_test
	X_test = X_test.fillna(0)
	X_no_categ = X_test.drop(['lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero',
		'd1_hero','d2_hero','d3_hero','d4_hero','d5_hero'],1)
	heroes_test = [X_test['r1_hero'],X_test['r2_hero'],X_test['r3_hero'],X_test['r4_hero'],X_test['r5_hero'],
				X_test['d1_hero'],X_test['d2_hero'],X_test['d3_hero'],X_test['d4_hero'],X_test['d5_hero']]

	all_heroes = pandas.concat(heroes_test)

	max_id = 0
	for _id in all_heroes.unique():
		if _id > max_id:
			max_id = _id
		
	N = len(all_heroes.unique())

	X_pick = np.zeros((X_test.shape[0], max_id))

	for i, match_id in enumerate(X_test.index):
	    for p in xrange(5):
	        X_pick[i, X_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
	        X_pick[i, X_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

	X_concat = pandas.concat([X_no_categ,pandas.DataFrame(X_pick,index=X_no_categ.index)],axis=1)
	#print(X_no_categ.shape, X_pick.shape, X_concat.shape)

	X_scaled = scaler.fit_transform(X_concat)
	
	#predicting on test data
	f1 = open('data/71.txt','w')
	print('match_id,radiant_win',file=f1)
	res = clf_best.predict_proba(X_scaled)[:,1]
	min_val = 1
	max_val = 0
	for i, ind in enumerate(X_test.index):
		print(str(ind)+','+str(res[i]),file=f1)
		if res[i] < min_val:
			min_val = res[i]
		if res[i] > max_val:
			max_val = res[i]

	print(min_val,max_val)

if __name__ == "__main__": 
	data = pandas.read_csv('features.csv',index_col='match_id')
	data_test = pandas.read_csv('features_test.csv',index_col='match_id')

	#boosting_task(data)
	#regression_task(data,data_test)

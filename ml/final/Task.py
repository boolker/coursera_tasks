import numpy as np
import pandas
import matplotlib.pyplot as plt

##Preparing data
data = pandas.read_csv('features.csv')
X = data[data.columns[0:103]]
y = np.ravel(data[data.columns[104:105]])
print(y)
print '------- 1.1. Features with gasps -------'
col = []
i = 0
for it in X.count():
    if it != max(X.count()):
        col = np.append(col, data.columns[i])
    i = i + 1
print col
del col,i
print ''
##Filling gasps
X = X.fillna(value = 0)
print '------- 1.2. Goal variable -------'
print data.columns[104:105][0]
print ''
##Cross_validation
from sklearn import cross_validation
kf = cross_validation.KFold(len(X),n_folds=5,shuffle=True, random_state=42)

##Gradient boosting
print '------- 1.3. Gradient boosting -------'
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import datetime
clf_gbc = GradientBoostingClassifier(n_estimators=30, random_state=42)
start_time = datetime.datetime.now()
y_score = cross_validation.cross_val_predict(clf_gbc, X=X, y=y, cv=kf)
end_time = datetime.datetime.now()
time_gbc = end_time - start_time
print 'Time for running of gradient boosting:', time_gbc
score_gbc = metrics.roc_auc_score(y_true=y, y_score=y_score)
print 'The quality of gradient boosting:', score_gbc
del GradientBoostingClassifier, datetime, clf_gbc, start_time, y_score, end_time, metrics
print ''
##Scaler
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X)
X_scale = scale.fit_transform(X)

##Logistic regression
print '------- 2.1 Logistic regression -------'
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
import datetime
score_lr = []
C = (np.array(range(20),dtype=np.float) + 1)/20
print 'Finding optimal C...'
for it in C:
    clf_lr = LogisticRegression(C=it, random_state = 42)
    y_score = cross_validation.cross_val_predict(clf_lr, X=X_scale, y=y, cv=kf)
    score_lr = np.append(score_lr,metrics.roc_auc_score(y_true=y, y_score=y_score))
##plt.plot(C,score_lr)
C_optim = C[score_lr == np.max(score_lr)][0]
print 'Optimal parametr C:', C_optim
start_time = datetime.datetime.now()
clf_lr = LogisticRegression(C=C_optim, random_state=42)
y_score = cross_validation.cross_val_predict(clf_lr, X=X_scale, y=y, cv=kf)
end_time = datetime.datetime.now()
time_lr = end_time - start_time
print 'Time for running of logistic regression:', time_lr
score_lr = metrics.roc_auc_score(y_true=y, y_score=y_score)
print 'The quality of logistic regression:', score_lr
del LogisticRegression, clf_lr, start_time, y_score, end_time, metrics, datetime, C
print ''
##Comparing algoritms
print '------- Comparing algorithms -------'
if score_lr > score_gbc:
    print 'Logistic regression is better!'
elif score_lr < score_gbc:
    print 'Gradient boosting is better!'
else:
    print 'The same quality for both algorithms.'        
if time_lr < time_gbc:
    print 'Logistic regression is faster!'
elif time_lr > time_gbc:
    print 'Gradient boosting is faster!'
else:
    print 'Both algorithms have equal running time.'
del score_lr, score_gbc, time_lr, time_gbc
print ''
    
#Delete categorical features
print '------- 2.2. Logistic regression (after deleting categorical features) -------'
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
clf_lr = LogisticRegression(C=C_optim, random_state=42)
features_del = ['lobby_type', 'r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
columns_without_hero = X.columns
X_heroes = X[features_del[1:]]
for it in features_del:
    columns_without_hero = columns_without_hero[columns_without_hero != it]
X = X[columns_without_hero]
X_scale = scale.fit_transform(X)
y_score = cross_validation.cross_val_predict(clf_lr, X=X_scale, y=y, cv=kf)
score_lr_new = metrics.roc_auc_score(y_true=y, y_score=y_score)
print 'The quality of logistic regression after deleting columns with information about heroes:', score_lr_new
del features_del, columns_without_hero, X, y_score, score_lr_new
print ''
#Quantity of DOTA heroes
print '------- 2.3. The number of identificators --------'
print 'The maximum identificator of the DOTA Hero:', max(pandas.unique(np.ravel(X_heroes)))
print 'The number of DOTA heroes:', len(pandas.unique(np.ravel(X_heroes)))
if max(pandas.unique(np.ravel(X_heroes)))>len(pandas.unique(np.ravel(X_heroes))):
    print 'The maximum identificator and the number are not equal! So some heroes are not available for picking.'
X_pick = np.zeros((data.shape[0], max(pandas.unique(np.ravel(X_heroes)))))
print ''
print '------- 2.4. Logistic regression with Bag of Words -------'
for i, match_id in enumerate(data.index):
    for p in xrange(5):
        X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X_scale = np.hstack((X_scale,X_pick))
score_lr = []
C = (np.array(range(20),dtype=np.float) + 1)/20
for it in C:
    clf_lr = LogisticRegression(C=it, random_state=42)
    y_score = cross_validation.cross_val_predict(clf_lr, X=X_scale, y=y, cv=kf)
    score_lr = np.append(score_lr,metrics.roc_auc_score(y_true=y, y_score=y_score))
C_optim = C[score_lr == np.max(score_lr)][0]
clf_lr = LogisticRegression(C=C_optim, random_state=42)
clf_lr.fit(X=X_scale, y=y)
print 'Optimal C after changing features:', C_optim
print 'The best quality after changing features with C=' + str(C_optim) + ':', np.max(score_lr)
del X_pick, y_score, score_lr, C, metrics, LogisticRegression, y, kf, data, X_scale
print ''
##C_optim = 0.3
##Checking on test data
print '------- 2.5. Checking on test data -------'
data = pandas.read_csv('features.csv')
X_train = data[data.columns[0:103]]
y_train = np.ravel(data[data.columns[104:105]])
X_train = X_train.fillna(value = 0)
del data

features_del = ['lobby_type', 'r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
columns_without_hero = X_train.columns
X_heroes = X_train[features_del[1:]]
X_pick = np.zeros((X_train.shape[0], max(pandas.unique(np.ravel(X_heroes)))))
for i, match_id in enumerate(X_train.index):
    for p in xrange(5):
        X_pick[i, X_train.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_train.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
del X_heroes
for it in features_del:
    columns_without_hero = columns_without_hero[columns_without_hero != it]
X_train = X_train[columns_without_hero]
del columns_without_hero

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X_train)
X_scale = scale.fit_transform(X_train)
del StandardScaler

X_train = np.hstack((X_scale,X_pick))
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression(C=C_optim, random_state=42)
clf_lr.fit(X=X_train, y=y_train)

X_test = pandas.read_csv('features_test.csv')
features_del = ['lobby_type', 'r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero']
columns_without_hero = X_test.columns
X_heroes = X_test[features_del[1:]]
X_test = X_test.fillna(value = 0)
X_pick = np.zeros((X_test.shape[0], max(pandas.unique(np.ravel(X_heroes)))))
for i, match_id in enumerate(X_test.index):
    for p in xrange(5):
        X_pick[i, X_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
del X_heroes

for it in features_del:
    columns_without_hero = columns_without_hero[columns_without_hero != it]
X_test = X_test[columns_without_hero]
del columns_without_hero

X_scale = scale.fit_transform(X_test)
X_test = np.hstack((X_scale,X_pick))
del scale
estim = clf_lr.predict_proba(X_test)
estim_radiant_win = estim[:,1]
print 'Maximum estimation:', max(estim_radiant_win)
print 'Minimum estimation:', min(estim_radiant_win)

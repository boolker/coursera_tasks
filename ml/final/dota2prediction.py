# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from  sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import KFold

from sklearn.metrics import roc_auc_score

X = pd.read_csv('features.csv',
                index_col='match_id')

y = X['radiant_win'].values  # целевая переменная

X = X.drop(['duration',
            'radiant_win',
            'tower_status_radiant',
            'tower_status_dire',
            'barracks_status_radiant',
            'barracks_status_dire'], axis=1)  # удаляем "пост-гейм" признаки

"""
# находим признаки с пропущенными значениями

features_count = X.count()
na_attributes = features_count.loc[
    features_count < features_count['start_time']]  # выбор атрибутов с пропущенными значениями

'''
Получили 12 признаков, в которых имеются пропущенные значения:

first_blood_time               77677
first_blood_team               77677
first_blood_player1            77677
first_blood_player2            53243
radiant_bottle_time            81539
radiant_courier_time           96538
radiant_flying_courier_time    69751
radiant_first_ward_time        95394
dire_bottle_time               81087
dire_courier_time              96554
dire_flying_courier_time       71132
dire_first_ward_time           95404

Поскольку данные содержат признаки, формируемые за 5 игровых минут,
соответствующие события за отведенное время могут просто не успеть
произойти. Например, если "первая кровь" (first_blood_time) произошла
на шестой минуте игры или позже. Аналогичная логика и для атрибута
"Время первого приобретения командой предмета "bottle" (dire_bottle_time).
'''
"""

X = X.fillna(0)  # заполняем пропущенные значения нулем

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=111)  # создаем разбиения для кросс-валидации

"""
# находим метрику для метода градиентного бустинга

X = X.values

for est in [10, 20, 30]:

    score = np.array([])

    start_time = dt.datetime.now()

    for train_index, test_index in kf:  # taking train and test indices from KFold

        # X - objects-attributes matrix, y - target variable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # fitting model
        clf = GradientBoostingClassifier(n_estimators=est, random_state=111)
        clf.fit(X_train, y_train)

        # predicting and calculating ROC_AUC
        score = np.append(score, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

    print 'Finished for ' + str(est) + ' trees'
    print 'ROC-AUC score is ' + str(score.mean())
    print 'Elapsed time: ' + str(dt.datetime.now() - start_time) + '\n'

'''
Результаты для метода градиентного бустинга:

Finished for 10 trees
ROC-AUC score is 0.665262596123
Elapsed time: 0:01:13.378777

Finished for 20 trees
ROC-AUC score is 0.682032297206
Elapsed time: 0:02:25.827041

Finished for 30 trees
ROC-AUC score is 0.689650274765
Elapsed time: 0:03:39.739869

Качество, скорее всего, продолжит расти, но очень медленно.
'''
"""

"""
# находим метрику для логистической регрессии

X = StandardScaler().fit_transform(X)  # нормировка

for c in [0.001, 0.01, 0.1, 1, 10]:

    score = np.array([])

    start_time = dt.datetime.now()

    for train_index, test_index in kf:  # taking train and test indices from KFold

        # X - objects-attributes matrix, y - target variable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # fitting model
        clf = LogisticRegression(C=c, random_state=111, penalty='l2')
        clf.fit(X_train, y_train)

        # predicting and calculating ROC_AUC
        score = np.append(score, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

    print 'Finished for C=' + str(c)
    print 'ROC-AUC score is ' + str(score.mean())
    print 'Elapsed time: ' + str(dt.datetime.now() - start_time) + '\n'

'''
Результаты для логистической регрессии (атрибуты нормированы):

Finished for C=0.001
ROC-AUC score is 0.71619776417
Elapsed time: 0:00:09.521735

Finished for C=0.01
ROC-AUC score is 0.716393394595
Elapsed time: 0:00:12.609828

Finished for C=0.1
ROC-AUC score is 0.716373924711
Elapsed time: 0:00:13.998484

Finished for C=1
ROC-AUC score is 0.7163711723
Elapsed time: 0:00:14.475044

Finished for C=10
ROC-AUC score is 0.716370688254
Elapsed time: 0:00:12.926181

Видим, что изменение коэффициента регуляризации влияет на метрику
лишь в пятом знаке.

Остановимся на параметре С=0.01
Наилучшее качество: 0.716393394595 (в модели градиентного
бустинга 0.689650274765).
Регрессия работает на порядок быстрее - 12 сек. против 2 мин.
'''
"""

"""
# находим метрику для логистической регрессии после удаления категориальных признаков

X = X.drop(['lobby_type',
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero',], axis=1)

X = StandardScaler().fit_transform(X)  # normalizing attributes

for c in [0.001, 0.01, 0.1, 1, 10]:

    score = np.array([])

    start_time = dt.datetime.now()

    for train_index, test_index in kf:  # taking train and test indices from KFold

        # X - objects-attributes matrix, y - target variable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # fitting model
        clf = LogisticRegression(C=c, random_state=111, penalty='l2')
        clf.fit(X_train, y_train)

        # predicting and calculating ROC_AUC
        score = np.append(score, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

    print 'Finished for C=' + str(c)
    print 'ROC-AUC score is ' + str(score.mean())
    print 'Elapsed time: ' + str(dt.datetime.now() - start_time) + '\n'

'''
Результаты логистической регрессии матрицы объекты-признаки БЕЗ категориальных признаков:

Finished for C=0.001
ROC-AUC score is 0.716192364184
Elapsed time: 0:00:07.987448

Finished for C=0.01
ROC-AUC score is 0.716397138427
Elapsed time: 0:00:09.937285

Finished for C=0.1
ROC-AUC score is 0.716376660688
Elapsed time: 0:00:11.298694

Finished for C=1
ROC-AUC score is 0.716373633801
Elapsed time: 0:00:11.052441

Finished for C=10
ROC-AUC score is 0.7163734018
Elapsed time: 0:00:11.516744


Лучший параметр С=0.01
Лучшее значение метрики 0.716397138427
(до исключения категориальных признаков 0.716393394595)

После исключения метрика улучшилась лишь в шестом знаке.
Возможно, на результат влияет не столько выбор героев,
сколько навык самих игроков.
'''
"""

"""
# нахождение количества уникальных героев

heroes = X.loc[:,['r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero']].values

unique_heroes = np.unique(heroes)


'''
В выборке 112 уникальных героев, их id:

[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
  19  20  21  22  23  25  26  27  28  29  30  31  32  33  34  35  36  37
  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55
  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73
  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91
  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 109 110 112]

'''
"""

"""
# кодируем "мешок слов" и находим метрику

X_pick = np.zeros((X.shape[0], 112))

for i, match_id in enumerate(X.index):
    for p in xrange(5):
        X_pick[i, X.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

X = X.drop(['lobby_type',
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero',], axis=1).values

X = np.concatenate([X,X_pick], axis=1)

X = StandardScaler().fit_transform(X)

for c in [0.001, 0.01, 0.1, 1, 10]:

    score = np.array([])

    start_time = dt.datetime.now()

    for train_index, test_index in kf:  # taking train and test indices from KFold

        # X - objects-attributes matrix, y - target variable
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # fitting model
        clf = LogisticRegression(C=c, random_state=111, penalty='l2')
        clf.fit(X_train, y_train)

        # predicting and calculating ROC_AUC
        score = np.append(score, roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

    print 'Finished for C=' + str(c)
    print 'ROC-AUC score is ' + str(score.mean())
    print 'Elapsed time: ' + str(dt.datetime.now() - start_time) + '\n'

'''
Результаты с закодированным мешком слов:

Finished for C=0.001
ROC-AUC score is 0.751406846699
Elapsed time: 0:00:15.353161

Finished for C=0.01
ROC-AUC score is 0.751767482447
Elapsed time: 0:00:20.271854

Finished for C=0.1
ROC-AUC score is 0.75174181764
Elapsed time: 0:00:22.178221

Finished for C=1
ROC-AUC score is 0.751734735383
Elapsed time: 0:00:23.399073

Finished for C=10
ROC-AUC score is 0.751733696309
Elapsed time: 0:00:21.363846

После кодирования результаты остались примерно теми же.
'''
"""

"""
# предсказываем значения для лучшей метрики Kaggle Score: 0.75526

X_pick = np.zeros((X.shape[0], 112))

for i, match_id in enumerate(X.index):
    for p in xrange(5):
        X_pick[i, X.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, X.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X = X.drop(['lobby_type',
            'r1_hero',
            'r2_hero',
            'r3_hero',
            'r4_hero',
            'r5_hero',
            'd1_hero',
            'd2_hero',
            'd3_hero',
            'd4_hero',
            'd5_hero'], axis=1).values

X_test = pd.read_csv('features_test.csv',
                     index_col='match_id')
match_id_ = X_test.index.values
X_test = X_test.fillna(0)

X_pick_test = np.zeros((X_test.shape[0], 112))

for i, match_id in enumerate(X_test.index):
    for p in xrange(5):
        X_pick_test[i, X_test.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick_test[i, X_test.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_test = X_test.drop(['lobby_type',
                      'r1_hero',
                      'r2_hero',
                      'r3_hero',
                      'r4_hero',
                      'r5_hero',
                      'd1_hero',
                      'd2_hero',
                      'd3_hero',
                      'd4_hero',
                      'd5_hero'], axis=1).values

X = np.concatenate([X, X_pick], axis=1)
X_test = np.concatenate([X_test, X_pick_test], axis=1)

SS = StandardScaler()
X = SS.fit_transform(X)  # normalizing attributes
X_test = SS.transform(X_test)

# fitting model
clf = LogisticRegression(C=0.01, random_state=111, penalty='l2')
clf.fit(X, y)

# predicting and calculating ROC_AUC
y_pred = clf.predict_proba(X_test)[:, 1]

sub = open('submit.csv', 'w')

sub.write('match_id,radiant_win' + '\n')

for i in range(0, match_id_.size):
    sub.write(str(match_id_[i]) + ',' + str(y_pred[i]) + '\n')

sub.close()

print 'Max prob:', y_pred.max()
print 'Min prob:', y_pred.min()
"""
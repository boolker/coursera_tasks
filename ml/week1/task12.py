from __future__ import print_function
import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
total = len(data)

subdata = data[['Pclass','Fare','Sex','Age','Survived']]

#print(subdata)

subdata['Sex'] = subdata.apply(lambda x: x[2]=='male',axis='columns')

subdata = subdata[np.isnan(subdata['Age'])!=True]
#print(subdata)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(subdata[['Pclass','Fare','Sex','Age']], subdata[['Survived']])
importances = clf.feature_importances_
print(importances)

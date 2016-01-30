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
f1 = open('data/121.txt','w')
sex = data['Sex'].value_counts()
#print(data['Sex'].value_counts())
print('1. ',sex['male'],' ',sex['female'])
print(sex['male'],' ',sex['female'], file=f1)
'''
f2 = open('data/112.txt','w')
survived = data['Survived'].value_counts()
print('2. ','{0:.2f}'.format(survived[1]*100/float(total)))
print('{0:.2f}'.format(survived[1]*100/float(total)),file=f2)


#print(data['Pclass'].value_counts())
f3 = open('data/113.txt','w')
classes = data['Pclass'].value_counts()
print('3. ', '{0:.2f}'.format(classes[1]*100/float(total)))
print('{0:.2f}'.format(classes[1]*100/float(total)),file=f3)

f4 = open('data/114.txt','w')
print('4. ', '{0:.2f}'.format(data['Age'].mean()),' ','{0:.2f}'.format(data['Age'].median()))
print(data['Age'].mean(),' ',data['Age'].median(),file=f4)

#print(data['SibSp'])
f5 = open('data/115.txt','w')
print('5. ', data['SibSp'].corr(data['Parch']))
print(data['SibSp'].corr(data['Parch']),file=f5)

f6 = open('data/116.txt','w')
names = []
f_names = data[data['Sex']=='female']['Name']
for name in f_names:
	#print(name)
	first_name = ''
	f_name = ((name.split(','))[1].strip().split('.'))	
	if f_name[0] == 'Miss':
		first_name = (f_name[1].strip().split(' '))[0]
	elif f_name[0] == 'Mrs':
		#print((f_name[1].strip().split('(')))
		if '(' in f_name[1]:
			name_parts = (f_name[1].strip().split('('))[1].split(' ')
			if len(name_parts) > 0:
				first_name = name_parts[0]
			
	#print(name, '-->',f_name[1], '-->', first_name)	
	if len(first_name) > 0:
		names.append(first_name.replace('(','').replace(')',''))
print(sorted(names))
df_names = pandas.DataFrame(names,columns=['name'])
top_names = df_names['name'].value_counts()
print(top_names)

print('6. ', top_names.axes[0][0])
print(top_names.axes[0][0],file=f6)
#print((df_names['name'].value_counts()),file=f6)'''
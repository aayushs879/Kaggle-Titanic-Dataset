# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 19:32:02 2018

@author: aayush
"""

#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing Dataset
dataset = pd.read_csv('train.csv')


y = dataset.iloc[:,1].values

#Dropping features
dataset['Cabin'] = dataset['Cabin'].fillna('A')
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset = dataset.drop([ 'Name', 'Ticket', 'PassengerId', ], axis = 1)
dataset = dataset.drop(['Embarked'], axis = 1)
dataset = dataset.drop(['Fare'], axis=1)
dataset = dataset.drop(['Cabin'], axis = 1)
X = dataset.iloc[:,1:].values

#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder,

encode2 = LabelEncoder()
X[:,1] = encode2.fit_transform(X[:,1])


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)


#Classifier
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', max_iter = -1, C = 30)


#For parameter tuning
from sklearn.model_selection import cross_val_score
cv = cross_val_score(estimator = clf, X=X, y=y, cv = 4)



#making predictions on submission script
test_data = pd.read_csv('test.csv')

test_data = test_data.drop(['Name', 'Ticket', 'PassengerId', 'Embarked', 'Fare', 'Cabin'], axis = 1)

test_data['Age']= test_data['Age'].fillna(dataset['Age'].mean())

X_test = test_data.iloc[:,:].values
X_test = scale.transform(X_test)

X_test[:,1] = encode1.transform(X_test[:,1])

clf.fit(X,y)
pred = clf.predict(X_test)
df = pd.DataFrame(pred)

submission = pd.concat([test_data.iloc[:,0],df], axis = 1)

pd.DataFrame.to_csv(self = submission, path_or_buf = 'C:\Aayush\Machine Learning\submission_titanic.csv') 


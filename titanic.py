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


#data preprocessing
for i in range(len(dataset['Fare'])):
  dataset['Fare'][i] = dataset['Fare'][i]+2
  
dataset['Fare']= np.log(dataset['Fare'])

#Dropping features
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset = dataset.drop([ 'Name', 'Ticket', 'PassengerId', ], axis = 1)
dataset = dataset.drop(['Embarked'], axis = 1)
dataset = dataset.drop(['Cabin'], axis = 1)
X = dataset.iloc[:,1:].values

#Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder

encode2 = LabelEncoder()
encode2.fit(X[:,1])
X[:,1] = encode2.fit_transform(X[:,1])


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)
    


#Classifier
from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', max_iter = -1, C = .6)


#For parameter tuning
from sklearn.model_selection import cross_val_score
cv = cross_val_score(estimator = clf, X=X, y=y, cv = 5)

accuracy = cv.mean()


#making predictions on submission script
test_data = pd.read_csv('test.csv')

test_data = test_data.drop(['Name', 'Ticket', 'PassengerId', 'Embarked', 'Cabin'], axis = 1)

test_data['Age']= test_data['Age'].fillna(dataset['Age'].mean())
for a in range(len(test_data['Fare'])):
  test_data['Fare'][a] = test_data['Fare'][a] + 2 #To avoid infinite values
test_data['Fare'] = np.log(test_data['Fare'])
test_data['Fare']= test_data['Fare'].fillna(test_data['Fare'].mean())

X_test = test_data.iloc[:,:].values
X_test = scale.transform(X_test)

X_test[:,1] = encode2.fit_transform(X_test[:,1])

clf.fit(X,y)
pred = clf.predict(X_test)
df = pd.DataFrame(pred)

submission = pd.concat([test_data.iloc[:,0],df], axis = 1)

pd.DataFrame.to_csv(self = submission, path_or_buf = 'C:\Aayush\Machine Learning\submission_titanic.csv') 


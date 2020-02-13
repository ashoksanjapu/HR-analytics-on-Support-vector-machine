# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:51:07 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
svm = pd.read_csv("C:\\Users\\User\\Downloads\\HRDataset.csv")

svm.columns
svm = svm.iloc[:,2:]

#visualization
sns.countplot(y='EmpStatusID',data=svm, palette='hls')
plt.show()
sns.countplot(y='DeptID',data=svm, palette='hls')
plt.show()
sns.countplot(y='PerfScoreID',data=svm, palette='hls')
plt.show()
sns.countplot(y='FromDiversityJobFairID',data=svm, palette='hls')
plt.show()
sns.countplot(y='Termd',data=svm, palette='hls')
plt.show()
sns.countplot(x='PositionID',data=svm, palette='hls')
plt.show()
plt.hist(svm['PayRate'])

# data preprocessing
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
svm.iloc[:, 7] = labelencoder_X.fit_transform(svm.iloc[:, 7])
svm.iloc[:, 8] = labelencoder_X.fit_transform(svm.iloc[:, 8])
svm.iloc[:, 9] = labelencoder_X.fit_transform(svm.iloc[:, 9])
x = svm.iloc[:,0:9]
y = svm.iloc[:,9]


# splitting the data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Create SVM classification object 
from sklearn.svm import SVC
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(x_train,y_train)
pred_test_linear = model_linear.predict(x_test)

np.mean(pred_test_linear==y_test)  #accuracy = 1.0

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(x_train,y_train)
pred_test_poly = model_poly.predict(x_test)

np.mean(pred_test_poly==y_test) #accuracy = 0.81

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(x_train,y_train)
pred_test_rbf = model_rbf.predict(x_test)

np.mean(pred_test_rbf==y_test) # Accuracy = 0.80

##linear model is best fit model compare to all other models because it has high accuracy
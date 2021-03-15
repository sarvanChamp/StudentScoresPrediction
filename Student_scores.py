# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:46:40 2021

@author: ELCOT
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("D:\datasets\student_scores.csv")
x=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,1].values

#for spliting data into train and test use sklearn

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.50,random_state=0)

#building model
from sklearn.linear_model import LinearRegression
regress=LinearRegression()
regress.fit(x_train,y_train)

#prediction
u_pred=regress.predict(x_test)

#Creating VisualRepresentation
plt.title('Simple Linear Regression')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regress.predict(x_train),color='blue')
plt.show()

plt.title('Simple Linear Regression with TestData')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,regress.predict(x_test),color='blue')
plt.show()

plt.title('Simple Linear Regression with TestData and TrainData')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regress.predict(x_train),color='blue')
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:17:11 2021

@author: hp
"""
#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import pickle



cal = pd.read_csv('calories.csv')
exc = pd.read_csv('exercise.csv')

#printing top rows in dataset
cal.head()
exc.head()

#Merging two datasets
data = pd.concat([exc,cal['Calories']], axis=1)
data.head()

#showing some stats about data
data.describe()

#shape of data (rows,cols)
data.shape

#Converting Categorical Data to Numerical Data 
data.replace({'Gender':{'male':0,'female':1}},inplace=True)

#Allocating features and target columns respectively to X & y variables
X = data.drop(columns=['User_ID','Calories'],axis=1)
y = data['Calories']

#Applying train_test_split method to divide data into train set & test set 
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2)

#creating RandomForestRegressor Model
xgb_model = XGBRegressor()

#Fitting Data to model
xgb_model.fit(X_train,y_train)

#Predicting data and store this prediction_result into variable
prediction = xgb_model.predict(X_test)

#Checking Performance 
print('Performance in training set-->',xgb_model.score(X_train,y_train))
print('Performance in testing set -->',xgb_model.score(X_test,y_test))

print(mean_absolute_error(y_test,prediction))

file = open('xgb_model.pkl','wb')
pickle.dump(xgb_model,file)




















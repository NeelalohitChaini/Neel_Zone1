import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Regression model can be built by using eitherstatsmodel or sklearn
## both has different pros and cons
import statsmodels
import statsmodels.api as sm
import sklearn
from sklearn.model_selection import train_test_split

# Read the given CSV file, and view some sample records
advertising = pd.read_csv("c:/Users/neela/Desktop/Pyhon/advertising.csv")
print(advertising.head())

#Let's inspect the various aspect of our dataframe
print(advertising.shape)
print(advertising.info())
print(advertising.describe())

## steps in model building
## 1. create X and y
## 2. create train and test sets (Rtios used 70-30,80-20)
## 3. Train your model on the training set (i.e. learn the coefficients)
## 4. Evaluate the model (training set, test set)

# create X and y
X = advertising['TV']
y = advertising['Sales']

# train-test-split this gives following 4 values
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70,random_state=100) #0.70 - means 70-30
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape) ## Original record count 200. 70% of will come 140

# training the model
## Stats model does not come with c (predictor) (y=mx+c) hence following adds
X_train_sm = sm.add_constant(X_train) 
print(X_train_sm)

##fitting the mode (giving the traimnig set X_train and y_train)
## OLS - ORDINARY LIST SUQARES
lr = sm.OLS(y_train,X_train_sm) #3 This cvreates a linear regression object
lr_model = lr.fit()
print(lr_model.params)

## summary method will give lot of other information
print(lr_model.summary())
 
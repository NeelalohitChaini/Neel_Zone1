import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Regression model can be built by using either statsmodel or sklearn
## both has different pros and cons
## Statsmodel gives more detailed information and required for Data scientists
## sklearn is used mostly by technology guys
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Read the given CSV file, and view some sample records
advertising = pd.read_csv("c:/Users/neela/Desktop/Pyhon/advertising.csv")
#print(advertising.head())

#Let's inspect the various aspect of our dataframe
#print(advertising.shape)
#print(advertising.info())
#print(advertising.describe())

# create X and y
X = advertising['TV']
y = advertising['Sales']

# train-test-split this gives following 4 values
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70,random_state=100) #0.70 - means 70-30
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape) ## Original record count 200. 70% of will come 140

## steps in sklearn model building 
# 1. Create an object of linear regression
# 2. fit the model
# 3. see the params , make predictions (trains,test)
# 4. Evaluate (r2, etc)

#. 1. Create an object of linear regression
lm = LinearRegression()

# 2. fit the model
#lm.fit(X_train,y_train) ## Only this will give error ValueError: Expected 2D array, got 1D array instead:
## so need to follow the nuance below

X_train.shape ## This is (140,) so it'sd a series so sklearn expects array and it wantts (140,1)
print(X_train.shape) 
## so we have to do reshaping
##reshape (140,) to (140,1)
X_train_lm = X_train.values.reshape(-1,1) ##-1 means numpy will take the total number of rows
X_test_lm = X_test.values.reshape(-1,1)
print(X_train_lm.shape)
lm.fit(X_train_lm,y_train) 

#. 3. see the params , make predictions (trains,test)
print(lm.coef_)
print(lm.intercept_)

#. make predictions (trains,test)
y_train_pred = lm.predict(X_train_lm)
y_test_pred = lm.predict(X_test_lm)

# 4. Evaluate (r2, etc)
print(r2_score(y_true=y_train,y_pred=y_train_pred))
print(r2_score(y_true=y_test,y_pred=y_test_pred))


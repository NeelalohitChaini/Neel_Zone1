import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Regression model can be built by using either statsmodel or sklearn
## both has different pros and cons
## Statsmodel gives more detailed information and required for Data scientists
## sklearn is used mostly by technology guys
import statsmodels
import statsmodels.api as sm
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Read the given CSV file, and view some sample records
advertising = pd.read_csv("c:/Users/neela/Desktop/Pyhon/advertising.csv")
#print(advertising.head())

#Let's inspect the various aspect of our dataframe
#print(advertising.shape)
#print(advertising.info())
#print(advertising.describe())

## steps in model building
## 1. Reading and UNnerstanding the Data
##   1.1.  create X and y
##   1.2.  create train and test sets (Rtios used 70-30,80-20)
## 2. Training the Model
##   2.1. Train your model on the training set (i.e. learn the coefficients)
##   2.2. Evaluate the model (training set, test set)
## 3. Residual Analysis
## 4. Predicting and evaluating on Test Set

# create X and y
X = advertising['TV']
y = advertising['Sales']

# train-test-split this gives following 4 values
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.70,random_state=100) #0.70 - means 70-30
#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape) ## Original record count 200. 70% of will come 140

# training the model
## Stats model does not come with c (predictor) (y=mx+c) hence following adds
X_train_sm = sm.add_constant(X_train) 
#print(X_train_sm)

##fitting the mode (giving the traimnig set X_train and y_train)
## OLS - ORDINARY LIST SUQARES
lr = sm.OLS(y_train,X_train_sm) #3 This cvreates a linear regression object
lr_model = lr.fit()
#print(lr_model.params)
  ## This has given the result 
    ## const    6.948683   ---> Intercept
    ## TV       0.054546   ---> Coefficient of TV
    ## Hence the model we have  Sales = 6.948683 + 0.054546 * TV

## summary method will give lot of other information
#print(lr_model.summary())

## Evaluate the model
plt.scatter(X_train,y_train) ## THis will give a scatter plot between X_train & y_train
#plt.show()

## Now with tyhis we can see what the model is predicting
## y_train is the actual value and the predicted value is y_train = 6.948683 + 0.054546 * X_train
## plt.plot(X_train, y_train) y_train = 6.948683 + 0.054546 * X_train

#plt.plot(X_train,6.948683 + 0.054546 * X_train ,'r') ## 'r' - will make the line in red colur
#plt.show() ## the blue points are the actual values and the points on the redline are the predicted y values

## The other way of doing instead of hadcoding the values
y_train_pred = lr_model.predict(X_train_sm)
#plt.plot(X_train,y_train_pred,'r') ## 'r' - will make the line in red colur
#plt.show()

##Step - 3  - Residual Analysis
## error = fn(y_train, y_train_pred)
res = y_train - y_train_pred
## plot the residuals
plt.figure()
sns.distplot(res)
plt.title("Residual Plot")
plt.show()
## Imp - look for patterns in the residual (we should not be able to identify)
plt.scatter(X_train,res)
plt.show()

##Step - 4 - Prediction and Evaluation on Test set
##  4.1. Make predictions on the test set (y_test_pred)
##  4.2. Evaluate the model, r-squared on the test set

## Stats model does not come with c (predictor) (y=mx+c) hence following adds
X_test_sm = sm.add_constant(X_test) 
y_test_pred = lr_model.predict(X_test_sm)

##Evaluate the model
## r-squared
r2 = r2_score(y_true=y_test,y_pred=y_test_pred)  ## THis will give 0.792103160125
print(r2)

## We can check the r-squared on the train set as well , whcih was doen using stats model
r2_train = r2_score(y_true=y_train,y_pred=y_train_pred) ## THis will give 0.815793313648 same as of stasmodel
print(r2_train)

## Inference - 0.792103160125 kis within 5% of 0.815793313648 so it shows stability. Means what the model
## has learnt on trainig set , it is able to generalize on the unseen test set as well

plt.scatter(X_test,y_test) ## THis will give a scatter plot between X_train & y_train
plt.plot(X_test,y_test_pred,'r') ## 'r' - will make the line in red colur
plt.show()
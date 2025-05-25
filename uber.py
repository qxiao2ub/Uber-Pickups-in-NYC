# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:09:45 2020

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.model_selection import train_test_split

# Import dataset from the csv file
os.chdir(r"C:\Users\44399\OneDrive - Office Everyday")
data = pd.read_csv("Uber-Jan-Feb-FOIL.csv")

# First do data exploration and make a scatter plot.
# Creat X and Y parameter first
X = data[['active_vehicles']]
Y = np.array(data[['trips']])

# Then plot a Scatter plot
plt.rc('font',family='STXihei',size=15)
plt.scatter(X,Y,60,color='blue',marker='o',linewidths=3,alpha=0.4)
plt.xlabel('active vehicles')
plt.ylabel('trips')
plt.title('Scatter plot')
plt.show()
    
# You can see that there is a clear linear positive correlation between active_vehicles and trips. 
# Let's build a linear model.
clf = linear_model.LinearRegression()
clf.fit(X,Y)

# Model results:
print('Intercept: {}'.format(clf.intercept_))
print('Coefficient: {}'.format(clf.coef_))
print("𝑅²: %s" % clf.score(X, Y))

# Plot the predicted linear and sacattered real data in the same picture.
plt.scatter(X,Y,color='green')
plt.plot(X,clf.predict(X), linewidth=3,color='blue')
plt.xlabel('active vehicles')
plt.ylabel('trips')
plt.show()

# Above, we used all the sample data to model and got the regression equation. 
# The scatter plot above shows that the model is ok.
# Next, we split the data set into training and test sets and see how well the model performs.

# Build the model
trian_X, test_X, train_Y, test_Y = train_test_split(X,Y, test_size=0.2, random_state=123)
regr1 = linear_model.LinearRegression()
regr1.fit(trian_X, train_Y)

print('Intercept: {}'.format(regr1.intercept_))
print('Coefficient: {}'.format(regr1.coef_))
# The results are slightly different from the original model.
# because we only used 80% of the data to model.

# Next, we use the test set to verify model performance.
# We use test_X to predict y and get y_pred1.
y_pred1 = regr1.predict(test_X)
print('Mean squared error:%.2f' % mean_squared_error(test_Y, y_pred1))
print('Variance score:%.2f' % r2_score(test_Y,y_pred1))
# Since the closer the Variance score is to 1, the better. 
# The variance score we got is 0.97, It means that the model works well.

# Try to improve the model (introduce a quadratic term x² into the model)
from sklearn. preprocessing import PolynomialFeatures 
ploy_reg = PolynomialFeatures(degree=2)
X_ = ploy_reg.fit_transform(X)
regr2 = linear_model.LinearRegression()
regr2.fit(X_, Y)

X2 = X.sort_values(['active_vehicles'])
X2_ = ploy_reg.fit_transform(X2)

# Plot to see how well the model fits:
plt.scatter(X,Y, color='green')
plt.plot(X2, regr2.predict(X2_), linewidth=3, color='blue')
plt.show()
# Plot looks the same as before

# Model test results:
y_pred2 = regr2.predict(ploy_reg.fit_transform(test_X))
print('Mean squared error: %.2f' % mean_squared_error(test_Y,y_pred2))
print('Variance score: %.2f' %r2_score(test_Y,y_pred2))
# We got a same Variance score as before, the model has not been further improved.

# Above are all using sklearn modeling, below we use statsmodels to model. 
# The output of modeling using this module is similar to R.
import statsmodels. api as sm 
X3 = sm.add_constant(X)
est = sm.OLS(Y, X3)
est2 =est.fit()
print('\n\n', est2.summary())


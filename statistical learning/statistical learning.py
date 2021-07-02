# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 22:33:13 2020

@author: Amin
"""

#### Statistical learning 
### Generating example regression data

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x + ss.uniform.rvs(loc=0,scale=1,size=n)

plt.figure()
plt.plot(x,y,'o',ms=5)
xx = np.array([0, 10])
plt.plot(xx, beta_0 + beta_1*xx)
plt.xlabel('X')
plt.ylabel('Y')
##
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# n = 100
# beta_0 = 5
# beta_1 = 2
# np.random.seed(1)
# x = 10 * ss.uniform.rvs(size=n)
# y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale = 1, size = n) 

###  Residual sum of squares 
def compute_rss(y_estimate, y):
  return sum(np.power(y-y_estimate, 2))

def estimate_y(x, b_0, b_1):
  return b_0 + b_1 * x

rss = compute_rss(estimate_y(x, beta_0, beta_1), y) 
###Least Squares Estimation in Code
rss = []
slopes = np.arange(-10, 15, 0.001)
for slope in slopes:
    rss.append(np.sum((y-beta_0-slope* x)**2))

ind_min = np.argmin(rss)
print('Estimate for the slope:', slopes[ind_min])

#plot figure
plt.figure()
plt.plot(slopes, rss)
plt.xlabel('Slope')
plt.ylabel('RSS')
### Simple linear regression in code
import statsmodels.api as sm
mod = sm.OLS(y,x)
est = mod.fit()
print(est.summary())
## considering constant
X = sm.add_constant(x)
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())

### scikit-learn for Linear Regression
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)

X = np.stack([x_1, x_2], axis=1)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_ylabel('$y$')

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True)
lm.fit(X,y)
lm.intercept_
lm.coef_[0]
lm.coef_[1]
X_0 = np.array([2,4])
lm.predict(X_0.reshape(1,-1))
lm.score(X,y)
### Assesing model accuracy
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.5, random_state=1)
lm = LinearRegression(fit_intercept = True)
lm.fit(X_train, y_train)
lm.score(X_test, y_test)
# =============================================================================
# 
# =============================================================================
### Generating Example Classification Data

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

h = 1 
sd = 1
n = 50

def gen_data(n,h,sd1,sd2):
    x_1 = ss.norm.rvs(-h, sd1, n)
    y_1 = ss.norm.rvs(0, sd1, n)
    
    x_2 = ss.norm.rvs(h, sd2, n)
    y_2 = ss.norm.rvs(0, sd2, n)
    return(x_1,y_1,x_2,y_2)

(x_1,y_1,x_2,y_2) = gen_data(50, 1, 1, 1.5)
(x_1,y_1,x_2,y_2) = gen_data(1000, 1.5, 1, 1.5)

def plot_data(x_1,y_1,x_2,y_2):
    plt.figure()
    plt.plot(x_1, y_1, 'o', ms=2)
    plt.plot(x_2, y_2, 'o', ms=2)
    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')
    
plot_data(x_1,y_1,x_2,y_2)
### Logistic Regression

from sklearn.linear_model import LogisticRegression
(x1,y1,x2,y2) = gen_data(1000, 1.5, 1, 1.5)
clf = LogisticRegression()
X = np.vstack((np.vstack((x1, y1)).T, np.vstack((x2, y2)).T))
X.shape
n = 1000
y = np.hstack((np.repeat(1,n), np.repeat(2,n)))
X_train,X_test,y_train,y_test = train_test_split(X, y, train_size=0.5, random_state=1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict_proba(np.array([-2, 0]).reshape(1, -1))
clf.predict(np.array([-2, 0]).reshape(1, -1))
## Computing Predictive Probabilities Across the Grid

def plot_probs(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    Z = probs[:,class_no]
    Z = Z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, Z)
    cbar = plt.colorbar(CS)
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")

plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_probs(ax, clf, 0)
plt.title("Pred. prob for class 1")
ax = plt.subplot(212)
plot_probs(ax, clf, 1)
plt.title("Pred. prob for class 2");
### Tree-Based Methods for Regression and Classification

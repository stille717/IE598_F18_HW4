#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:47:35 2018

@author: stille
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

df = pd.read_csv('/Users/stille/Desktop/UIUC/MachineLearning/HW4/housing.csv')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())

#draw statistical plots
sns.set(style='whitegrid', context='notebook')
cols = ['CRIM', 'ZN', 'INDUS', 'CHAS','NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5);
plt.show()

#heatmaps
cm = np.corrcoef(df[cols].values.T) 
sns.set(font_scale=1.5) 
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols, xticklabels=cols) 
plt.show() 

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = pd.DataFrame(self.net_input(X))
            errors = (y - output)
            self.w_[1] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return self.net_input(X)
    
X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values.reshape(-1,1)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(X)
sc_y.fit(y)

X_std = pd.DataFrame(X)
y_std = pd.DataFrame(y)

X_std = sc_x.transform(X_std)
y_std = sc_y.transform(y_std)


lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

#num_rooms_std = sc_x.transform([5])
#price_std = lr.predict(num_rooms_std)
#print("Price in $1000's: %.3f" % \sc_y.inverse_transform(price_std))

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



#Linear regression
print('Linear regression start:')
slr = LinearRegression()
slr.fit(X, y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)
lin_regplot(X, y, slr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
y_train_pred1 = slr.predict(X_train)
y_test_pred1 = slr.predict(X_test)
#residual errors
plt.scatter(y_train_pred1, y_train_pred1 - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred1, y_test_pred1 - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
#MSE
print('Linear regression MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred1),mean_squared_error(y_test, y_test_pred1)))
#R2
print('Linear regression R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred1),r2_score(y_test, y_test_pred1)))



#Ridge regression
print('Ridge regression start:')
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print('Slope: %.3f' % ridge.coef_[0])
print('Intercept: %.3f' % ridge.intercept_)
lin_regplot(X, y, ridge)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
y_train_pred2 = ridge.predict(X_train)
y_test_pred2 = ridge.predict(X_test)
#residual errors
plt.scatter(y_train_pred2, y_train_pred2 - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred2, y_test_pred2 - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
#MSE
print('Ridge regression MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred2),mean_squared_error(y_test, y_test_pred2)))
#R2
print('Ridge regression R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred2),r2_score(y_test, y_test_pred2)))
#Compare with different alpha
alpha_space = np.logspace(-4, 0, 50)
MSE_test_scores = []
MSE_train_scores = []
R2_test_scores = []
R2_train_scores = []
for alpha in alpha_space:
    ridge.alpha = alpha
    ridge.fit(X, y)
    y_train_pred2 = ridge.predict(X_train)
    y_test_pred2 = ridge.predict(X_test)
    MSE_train_scores.append(mean_squared_error(y_train, y_train_pred2))
    MSE_test_scores.append(mean_squared_error(y_test, y_test_pred2))
    R2_train_scores.append(r2_score(y_train, y_train_pred2))
    R2_test_scores.append(r2_score(y_test, y_test_pred2))
plt.plot(alpha_space, MSE_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE_train_scores')
plt.show()
plt.plot(alpha_space, MSE_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE_test_scores')
plt.show()
plt.plot(alpha_space, R2_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('R2_train_scores')
plt.show()
plt.plot(alpha_space, R2_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('R2_test_scores')
plt.show()


#Lasso regression
print('Lasso regression start:')
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
print('Slope: %.3f' % lasso.coef_[0])
print('Intercept: %.3f' % lasso.intercept_)
lin_regplot(X, y, lasso)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
y_train_pred3 = lasso.predict(X_train).reshape(361,1)
y_test_pred3 = lasso.predict(X_test).reshape(91,1)
#residual errors
plt.scatter(y_train_pred3, y_train_pred3 - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred3, y_test_pred3 - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
#MSE
print('Lasso regression MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred3),mean_squared_error(y_test, y_test_pred3)))
#R2
print('Lasso regression R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred3),r2_score(y_test, y_test_pred3)))
#Compare with different alpha
alpha_space = np.logspace(-4, 0, 50)
MSE_test_scores = []
MSE_train_scores = []
R2_test_scores = []
R2_train_scores = []
for alpha in alpha_space:
    lasso.alpha = alpha
    lasso.fit(X, y)
    y_train_pred3 = lasso.predict(X_train).reshape(361,1)
    y_test_pred3 = lasso.predict(X_test).reshape(91,1)
    MSE_train_scores.append(mean_squared_error(y_train, y_train_pred3))
    MSE_test_scores.append(mean_squared_error(y_test, y_test_pred3))
    R2_train_scores.append(r2_score(y_train, y_train_pred3))
    R2_test_scores.append(r2_score(y_test, y_test_pred3))
plt.plot(alpha_space, MSE_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE_train_scores')
plt.show()
plt.plot(alpha_space, MSE_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('MSE_test_scores')
plt.show()
plt.plot(alpha_space, R2_train_scores)
plt.xlabel('alpha_space')
plt.ylabel('R2_train_scores')
plt.show()
plt.plot(alpha_space, R2_test_scores)
plt.xlabel('alpha_space')
plt.ylabel('R2_test_scores')
plt.show()



#ElasticNet regression
print('ElasticNet regression start:')
elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic.fit(X, y)
print('Slope: %.3f' % elastic.coef_[0])
print('Intercept: %.3f' % elastic.intercept_)
lin_regplot(X, y, elastic)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()
y_train_pred4 = elastic.predict(X_train).reshape(361,1)
y_test_pred4 = elastic.predict(X_test).reshape(91,1)
#residual errors
plt.scatter(y_train_pred4, y_train_pred4 - y_train,c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred4, y_test_pred4 - y_test,c='lightgreen', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
plt.xlim([-10, 50])
plt.show()
#MSE
print('ElasticNet regression MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred4),mean_squared_error(y_test, y_test_pred4)))
#R2
print('ElasticNet regression R^2 train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred4),r2_score(y_test, y_test_pred4)))
#Compare with different l1_ratio
l1_ratio_space = np.logspace(-4, 0, 50)
MSE_test_scores = []
MSE_train_scores = []
R2_test_scores = []
R2_train_scores = []
for l1_ratio in l1_ratio_space:
    elastic.l1_ratio = l1_ratio
    elastic.fit(X, y)
    y_train_pred3 = elastic.predict(X_train).reshape(361,1)
    y_test_pred3 = elastic.predict(X_test).reshape(91,1)
    MSE_train_scores.append(mean_squared_error(y_train, y_train_pred3))
    MSE_test_scores.append(mean_squared_error(y_test, y_test_pred3))
    R2_train_scores.append(r2_score(y_train, y_train_pred3))
    R2_test_scores.append(r2_score(y_test, y_test_pred3))
plt.plot(l1_ratio_space, MSE_train_scores)
plt.xlabel('l1_ratio_space')
plt.ylabel('MSE_train_scores')
plt.show()
plt.plot(l1_ratio_space, MSE_test_scores)
plt.xlabel('l1_ratio_space')
plt.ylabel('MSE_test_scores')
plt.show()
plt.plot(l1_ratio_space, R2_train_scores)
plt.xlabel('l1_ratio_space')
plt.ylabel('R2_train_scores')
plt.show()
plt.plot(l1_ratio_space, R2_test_scores)
plt.xlabel('l1_ratio_space')
plt.ylabel('R2_test_scores')
plt.show()





print("My name is Wenyu Ni")
print("My NetID is: wenyuni2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

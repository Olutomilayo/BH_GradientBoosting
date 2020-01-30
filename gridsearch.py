import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import StackingRegressor
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import pickle
#from keras.models import Sequential
#from keras.layers import Dense

os.getcwd()
os.chdir('C:/Users/opetinrin2/OneDrive - City University of Hong Kong/CityU/test')

#data = pd.read_csv ('Scaled_dataset.csv')
#print(data)
#Y = data.iloc[:, 120]
#data=data[~Y.isin ([0])]
#data.dropna()
#data.replace([np.inf, -np.inf], np.nan, inplace=True)
#data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
#print(data)
#data.to_csv('ot.csv')

data = pd.read_csv ('ot.csv',sep=',',dtype= np.float64)


#print(data)

#data.replace([np.inf, -np.inf], np.nan, inplace=True)

data.dropna(inplace=True)

#data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
#print(data.shape)

# Split into X and Y
data=np.array(data)
print(data.shape)

X = data[:, 1:121]
Y = data[:, 121]

X = np.array(X)
Y = np.array(Y)

#print(X)
#print(Y)


#for index in range(100):
    #Split into test and train
index=99
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = index)

#Scaling of data
SS = StandardScaler()
X_train = SS.fit_transform(X_train)
X_test = SS.transform (X_test)

##    Principal Compponent Analysis
#    pca = KernelPCA()
#    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)

'''#Feature Selection
selector = SelectFromModel(estimator=Ridge()) 
selector.fit(X_train, Y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)'''

#RF
parameters = {'n_estimators': [100, 500, 1000], 'bootstrap': [True, False], 'random_state':[index], 'max_features': ['auto', 'sqrt', 'log2'], 'criterion' : ['mse', 'mae']}
rf = RandomForestRegressor()

RFR = GridSearchCV(rf, parameters,cv=10)
RFR.fit(X_train,Y_train)

Y_pred=RFR.predict(X_test)
print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")
print("\n The best estimator across ALL searched params:\n",
      RFR.best_estimator_)
print("\n The best score across ALL searched params:\n",
      RFR.best_score_)
print("\n The best parameters across ALL searched params:\n",
      RFR.best_params_)
print("\n ========================================================")

   
#SVR
#parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
#svc = SVR(gamma='auto')
#
#SupportVR = GridSearchCV(svc, parameters,cv=10)
#SupportVR.fit(X_train,Y_train)
#
#Y_pred=SupportVR.predict(X_test)
#print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
#print("\n========================================================")
#print(" Results from Grid Search " )
#print("========================================================")
#print("\n The best estimator across ALL searched params:\n",
#      RFR.best_estimator_)
#print("\n The best score across ALL searched params:\n",
#      RFR.best_score_)
#print("\n The best parameters across ALL searched params:\n",
#      RFR.best_params_)
#print("\n ========================================================")
#print('rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)))

#    XGB
#    parameters = {'objective' : ['reg:linear'], 'learning_rate' : [0, 1], 'n_estimators' : [100, 1000], 'gamma' : [0,2], 'max_depth' : [2, 10], 'colsample_bytree' : [0.1, 1], 'reg_alpha' : [0,1]}
#    xg_reg = xgb.XGBRegressor()
#    XGB = GridSearchCV(xg_reg, parameters,cv=10)
#    XGB.fit(X_train,Y_train)
#
#    Y_pred=XGB.predict(X_test)
#    print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)))

#GB
parameters = {'n_estimators': [100, 1000], 'loss': ['ls', 'lad', 'quantile', 'huber'], 'learning_rate':[0.1, 0.5, 1.0], 'min_samples_split': [2], 'max_depth': [5, 10, 20, 50], 'random_state': [index]}
gb = GradientBoostingRegressor()

GBR = GridSearchCV(gb, parameters,cv=10)
GBR.fit(X_train,Y_train)

Y_pred=GBR.predict(X_test)
print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")
print("\n The best estimator across ALL searched params:\n",
      GBR.best_estimator_)
print("\n The best score across ALL searched params:\n",
      GBR.best_score_)
print("\n The best parameters across ALL searched params:\n",
      GBR.best_params_)
print("\n ========================================================")
print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)))

#AB
parameters = {'n_estimators': [100, 1000], 'loss': ['linear', 'square', 'exponential'], 'learning_rate':[1.0, 5.0, 10.0], 'random_state': [index]}
ab = AdaBoostRegressor()

ABR = GridSearchCV(ab, parameters,cv=10)
ABR.fit(X_train,Y_train)

Y_pred=ABR.predict(X_test)
print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")
print("\n The best estimator across ALL searched params:\n",
      ABR.best_estimator_)
print("\n The best score across ALL searched params:\n",
      ABR.best_score_)
print("\n The best parameters across ALL searched params:\n",
      ABR.best_params_)
print("\n ========================================================")
print('rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)))

#Bagging
parameters = {'n_estimators': [100, 1000], 'max_samples': [500, 1000, 1500], 'max_features': [50, 75, 100], 'random_state': [index]}
bg = BaggingRegressor()

Bag = GridSearchCV(bg, parameters,cv=10)
Bag.fit(X_train,Y_train)

Y_pred=Bag.predict(X_test)
print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")
print("\n The best estimator across ALL searched params:\n",
      Bag.best_estimator_)
print("\n The best score across ALL searched params:\n",
      Bag.best_score_)
print("\n The best parameters across ALL searched params:\n",
      Bag.best_params_)
print("\n ========================================================")
print('random_state', str(index) ,'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)))

#Stacking
#parameters = {'estimators' : [SVR, KNeighborsRegressor, DecisionTreeRegressor], 'final_estimators': [RandomForestRegressor], 'cv': [10], 'random_state':[index]}
#stack = StackingRegressor()
#
#STK = GridSearchCV(stack, parameters,cv=10)
#STK.fit(X_train,Y_train)
#
#Y_pred=STK.predict(X_test)
#print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
#print("\n========================================================")
#print(" Results from Grid Search " )
#print("========================================================")
#print("\n The best estimator across ALL searched params:\n",
#      STK.best_estimator_)
#print("\n The best score across ALL searched params:\n",
#      STK.best_score_)
#print("\n The best parameters across ALL searched params:\n",
#      STK.best_params_)
#print("\n ========================================================")

#MLP
parameters = {'hidden_layer_sizes': [200,100], 'alpha':[0.0001, 0.001, 0.01, 0.1, 1], 'activation' : ['identity', 'tanh', 'logistic', 'relu'], 'solver' :['sgd', 'lbfgs', 'adam'], 'random_state' : [index], 'max_iter' : [200, 300, 400], 'learning_rate': ['constant']}
MLP = MLPRegressor()

NN = GridSearchCV(MLP, parameters,cv=10)
NN.fit(X_train,Y_train)

Y_pred=NN.predict(X_test)

print('random_state: ', str(index), 'rmse：', np.sqrt(mean_squared_error(Y_test, Y_pred)), 'r^2: ', r2_score(Y_test, Y_pred), 'mae: ', (mean_absolute_error(Y_test, Y_pred)))
print("\n========================================================")
print(" Results from Grid Search " )
print("========================================================")
print("\n The best estimator across ALL searched params:\n",
      NN.best_estimator_)
print("\n The best score across ALL searched params:\n",
      NN.best_score_)
print("\n The best parameters across ALL searched params:\n",
      NN.best_params_)
print("\n ========================================================")
    
    
    
with open('RFKPCA.pickle', "wb") as f:
    pickle.dump(RFR, f)
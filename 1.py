""" Assignment 3: EN4553 (Machine Vision) """
__author__ = "D.M.Asitha Indrajith Divisekara"
__indexNo__ = "170150A"

import numpy as np
from typing import Tuple
from numpy.core.fromnumeric import mean

from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error # this can be achieved using np.mean function

import matplotlib.pyplot as plt


# question 1 - part a
# read the dataset as a set of arrays.
def load_dataset(src_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    x_train = np.loadtxt(src_dir + 'x_train.txt')
    y_train = np.loadtxt(src_dir + 'y_train.txt')
    x_val = np.loadtxt(src_dir + 'x_val.txt')
    y_val = np.loadtxt(src_dir + 'y_val.txt')
    x_test = np.loadtxt(src_dir + 'x_test.txt')

    return x_train, y_train, x_val, y_val, x_test
#################################################################################################################################

# n - hyper parameter 
# each w is a weight
# variables x,y represent rows in x_train.txt y_train.txt
# question 1 part b sub part a
# make input features for the above linear regression model.
def get_features(x: np.ndarray, n: int) -> np.ndarray:
    features = []

    for i in range(1,n+1):
        features.append(np.power(x,i))

    features_nparray = np.array(features)

    features_output = np.transpose(features_nparray)

    return features_output
###################################################################################################################################    

# question 1 part b sub part a
# fit and evaluate function
def fit_and_evaluate(x_train:np.ndarray, y_train:np.ndarray, x_val:np.ndarray, y_val:np.ndarray, n:int) -> Tuple[float, float]:
    regressor = LinearRegression()
    x_train_features = get_features(x_train, n)
    x_val_features = get_features(x_val, n)
    regressor.fit(x_train_features,y_train)
    
    # W = regressor.coef_
    # print('W shape = ', W.shape)
    # W_transpose = np.transpose(W)
    # print('W_transpose shape = ', W_transpose.shape)
    
    y_predict_train = regressor.predict(x_train_features)
    y_predict_val = regressor.predict(x_val_features)

    # we can use numpy mean function to find the Mean Squared Error no need to import sklearn metrics
    # train_mse = mean_squared_error(y_train, y_predict_train) 
    train_mse = np.mean((y_predict_train - y_train)**2)

    # val_mse = mean_squared_error(y_val, y_predict_val)
    val_mse = np.mean((y_predict_val - y_val)**2)
    
    return train_mse, val_mse
    
####################################################################################################################################

# the datasets conaining path can be given wrt the root folder
Dataset_dir = "dataset/"

x_train, y_train, x_val, y_val, x_test = load_dataset(Dataset_dir)

train_mse_array = []
val_mse_array = []
for i in range(1,11):
    train_mse,val_mse = train_mse, val_mse = fit_and_evaluate(x_train, y_train, x_val, y_val,i)
    train_mse_array.append(train_mse)
    val_mse_array.append(val_mse)

# plotting the train and validation squarred errors
plt.plot(range(1,11), train_mse_array, "o-", color="red", label="Train MSE")
plt.plot(range(1,11), val_mse_array, "o-", color="green", label="Validation MSE")
plt.axhline(y = min(val_mse_array), color = 'black', linestyle = 'dashed')
plt.axvline(x = 1+val_mse_array.index(min(val_mse_array)), color = 'black', linestyle = 'dashed')

plt.xticks(range(1,11))
plt.yticks(list(np.linspace(0,0.08,5)) + [min(val_mse_array)])
plt.legend()
plt.xlabel('n hyperparameter range 1-10')
plt.ylabel('Train/validation  Mean Square Error')
plt.title('Hyper-parameter n vs train and validation MSE')

plt.savefig('Train_and_validation_MSE_for_n=1_to_10.png')
plt.show()




















""" Assignment 3: EN4553 (Machine Vision) """
__author__ = "D.M.Asitha Indrajith Divisekara"
__indexNo__ = "170150A"

import numpy as np
from typing import Tuple

# question 1 - part a
# read the dataset as a set of arrays.
def load_dataset(src_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    x_train = np.loadtxt(src_dir + '/x_train.txt')
    y_train = np.loadtxt(src_dir + '/y_train.txt')
    x_val = np.loadtxt(src_dir + '/x_val.txt')
    y_val = np.loadtxt(src_dir + '/y_val.txt')
    x_test = np.loadtxt(src_dir + '/x_test.txt')

    return x_train, y_train, x_val, y_val, x_test

Dataset_dir = "C:/Users/Asitha/Desktop/machine vision assignment 3/170150A/dataset"
x_train, y_train, x_val, y_val, x_test = load_dataset(Dataset_dir)
# print('x_train shape = ', x_train.shape, '\n')


# n - hyper parameter 
# each w is a weight
# variables x,y represent rows in x_train.txt y_train.txt
  
def get_features(x: np.ndarray, n: int) -> np.ndarray:
    features = []

    for i in range(1,n+1):
        features.append(np.power(x,i))

    features_nparray = np.array(features)

    features_output = np.transpose(features_nparray)

    return features_output
    
sample = np.array([1.0, 2.0, 3.0, 4.0])
features = get_features(sample, 3)
print(features.shape)














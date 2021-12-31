import numpy as np
from typing import Tuple

def load_dataset(src_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load and return the dataset

    x_train = np.loadtxt(src_dir + '/x_train.txt')
    y_train = np.loadtxt(src_dir + '/y_train.txt')
    x_val = np.loadtxt(src_dir + '/x_val.txt')
    y_val = np.loadtxt(src_dir + '/y_val.txt')
    x_test = np.loadtxt(src_dir + '/x_test.txt')

    return x_train, y_train, x_val, y_val, x_test

Dataset_dir = "C:/Users/Asitha/Desktop/machine vision assignment 3/170150A/dataset"
print(type(load_dataset(Dataset_dir)))


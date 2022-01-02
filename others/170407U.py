my_folder_path='170407U/'

import numpy as np
from typing import Tuple
import os

def load_dataset(src_dir)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  np_file_list=[]
  files_list=os.listdir(src_dir)
  for FILE in files_list:
    np_file=np.loadtxt(os.path.join(src_dir,FILE))
    np_file_list.append(np_file)
  return np_file_list[0], np_file_list[1], np_file_list[2], np_file_list[3], np_file_list[4]

x_train,y_train,x_val,y_val,x_test = load_dataset(my_folder_path)


def get_features(x,n) -> np.ndarray:
  x_dim=x.size
  x_ones=np.ones((x_dim,n))
  for i in range(x_dim):
    count=1
    for j in range(n):
      x_ones[i][j]=x[i]**count
      count+=1
  return np.array(x_ones)

x = get_features(np.array([1.0,2.0,3.0]),3)

x1 = get_features(np.array([1.0,2.0,3.0]),4)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib



def fit_and_evaluate(x_train, y_train,x_test, x_val, y_val, n)-> Tuple[float, float]:

  X_train=get_features(x_train,n)
  reg = LinearRegression()
  reg.fit(X_train, y_train)
  y_predict_train = reg.predict(X_train)
  train_mse = mean_squared_error(y_train, y_predict_train)

  # Generate model predictions for the val set and calculate the MSE.
  X_val = get_features(x_val, n)
  y_predict_val = reg.predict(X_val)
  val_mse = mean_squared_error(y_val, y_predict_val)

  # Generate model predictions for the test set and save the relavent text files.
  X_test = get_features(x_test,n)
  y_test_predict = reg.predict(X_test)
  np.savetxt('170407U/170407U_y_predict_test_2'+str(n)+'.txt', y_test_predict, fmt='%1.18e')

  return (train_mse, val_mse)

Train_MSE=[]
Val_MSE = []
for n in range(1,11):
  train_mse, val_mse = fit_and_evaluate(x_train, y_train,x_test, x_val, y_val, n)
  Train_MSE.append(train_mse)
  Val_MSE.append(val_mse)
  print("{}---> Train MSE: {}, Test MSE: {}\n".format(n, train_mse, val_mse))
  
import matplotlib.pyplot as plt

n=[i for i in range(1,11)]
plt.style.use('fivethirtyeight')
plt.scatter(n, Train_MSE, color = "r", marker = "o", s = 50, label="Train MSE")
plt.scatter(n, Val_MSE, color = "b",  marker = "o", s = 50, label="Train MSE")
plt.plot(n, Train_MSE, color="r")
plt.plot(n, Val_MSE, color="b")
plt.rcParams["figure.figsize"] = [30.00, 15.00]
plt.rcParams["figure.autolayout"] = True
plt.xlabel("n")
plt.ylabel("MSE")
plt.title("Train MSE and Val MSE against n")
plt.legend()
plt.xlim(-1, 11)
plt.ylim(-0.05, 0.4)
plt.grid(True)
plt.show()

plt.plot(range(1,11), Train_MSE, "o-", color="red", label="Train MSE")
plt.plot(range(1,11), Val_MSE, "o-", color="blue", label="Validation MSE")
plt.axhline(y = min(Val_MSE), color = 'black', linestyle = 'dashed')
plt.axvline(x = 1+Val_MSE.index(min(Val_MSE)), color = 'black', linestyle = 'dashed')

plt.xticks(range(1,11))
plt.yticks(list(np.linspace(0,0.4,20)) + [min(Val_MSE)])
plt.legend()
plt.xlabel('n hyperparameter range 1-10')
plt.ylabel('Train/Validation  Mean Square Error')
plt.title('Hyper-parameter n vs Train and Validation MSE')

plt.savefig('Train_and_validation_MSE_vs_n.png')
plt.show()

model = joblib.load('170407U/reg_7.joblib')  #n=9

X_test = get_features(x_test,7)

y_test_predict = model.predict(X_test)

np.savetxt('/170407U/170407U_y_predict.txt', y_test_predict, fmt='%1.18e')

def write_txt(new_data, filename):
    with open(filename, 'a+') as outfile:
        outfile.write(new_data)


for i in y_test_predict:
  write_txt(str(i), "170407U/y_test.txt")


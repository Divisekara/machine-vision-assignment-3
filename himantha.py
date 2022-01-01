import numpy as np
from typing import Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

#part a
def load_dataset(src_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  x_train=np.loadtxt(src_dir+"/x_train.txt")
  y_train=np.loadtxt(src_dir+"/y_train.txt")
  x_val=np.loadtxt(src_dir+"/x_val.txt")
  y_val=np.loadtxt(src_dir+"/y_val.txt")
  x_test=np.loadtxt(src_dir+"/x_test.txt")
  return tuple([x_train,y_train,x_val,y_val,x_test])
  

[x_train,y_train,x_val,y_val,x_test]=load_dataset("dataset2")
# plt.scatter(x_train,y_train,label='train_data')
# plt.scatter(x_val,y_val,label='validation_data')
# plt.xlabel=('x')
# plt.ylabel=('y')
# plt.legend()

def get_features(x: np.ndarray, n: int) -> np.ndarray:
  feature_matrix=np.zeros([x.shape[0],n])
  for j in range(n):
    feature_matrix[:,j]=np.power(x,j+1)
  return feature_matrix

from numpy.ma.core import shape
u=get_features(x_train,3)

feature_matrix=np.zeros([4,5])

feature_matrix[:,2]=[1,1,1,1]
print(feature_matrix)

LR=LinearRegression()
mse=mean_squared_error

def fit_and_evaluate(x_train: np.ndarray, y_train: np.ndarray,x_val: np.ndarray, y_val: np.ndarray,n: int) -> Tuple[float, float]:
   model = LR.fit(get_features(x_train,n), y_train)
   y_predict_train = model.predict(get_features(x_train,n))
   train_mse = mse(y_train, y_predict_train)


   model = LR.fit(get_features(x_val,n), y_val)
   y_predict_val = model.predict(get_features(x_val,n))
   val_mse = mse(y_val, y_predict_val)
   return train_mse, val_mse

print(fit_and_evaluate(x_train,y_train,x_val,y_val,3))

train_mse_values=[]
val_mse_values=[]
for i in range (0,10):
  [train_mse,val_mse]=fit_and_evaluate(x_train,y_train,x_val,y_val,i+1)
  train_mse_values.append(train_mse)
  val_mse_values.append(val_mse)

plt.plot(range(1,11),train_mse_values,label="train_mse")
plt.plot(range(1,11),val_mse_values,label='validation_mse')
plt.xlabel("n")
plt.ylabel("MSE")
plt.legend()
plt.show()

print(train_mse_values.index(min(train_mse_values))+1)
print(val_mse_values.index(min(val_mse_values))+1)

#part(b)
min_n=10
final_model = LR.fit(get_features(x_train,min_n), y_train)
y_test = final_model.predict(get_features(x_test,min_n))
#print(y_test)
text=''
for i in range(len(y_test)):
  line=y_test[i]
  text=text+str(line)+"\n"
text_file = open("/170416V_y_predict_test.txt", "w")
n = text_file.write(text)
text_file.close()
#print(text)
plt.scatter(x_test,y_test)
# plt.xlabel("x_test")
# plt.ylabel("y_test")
plt.show()










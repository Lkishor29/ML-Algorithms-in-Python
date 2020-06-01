import pandas as pd
import numpy as np

#Get the Dataset from https://www.kaggle.com/andonians/random-linear-regression/data

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
x_train=train['x']
x_test=test['x']
y_train=train['y']
y_test=test['y']


#Forming 2D Arrays
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(r2_score(y_test,y_pred))


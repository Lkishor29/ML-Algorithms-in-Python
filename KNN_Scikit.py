#KNN using Scikit
#hs_makkar

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x,y=load_digits(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)
#Accuracy
#print(model.score(x_test,y_test))


#How to select values for K
import matplotlib.pyplot as plt
neighbours=np.arange(1,15)
train_acc=np.empty(14)
test_acc=np.empty(14)
for i,k in enumerate(neighbours):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    
    train_acc[i]=knn.score(x_train,y_train)
    test_acc[i]=knn.score(x_test,y_test)
    

plt.plot(neighbours,test_acc,label='Test Accuracy')
plt.plot(neighbours,train_acc,label='Train Accuracy')

plt.legend()
plt.xlabel('N_Neighbours')
plt.ylabel('Accuracy')
plt.show()
    

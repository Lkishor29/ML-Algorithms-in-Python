#hs_makkar
#Using Inbuilt Library

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
x,y=load_digits(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
model=LogisticRegression(solver='lbfgs',multi_class='ovr',
                         random_state=0)

model.fit(x_train,y_train)
x_test=scaler.fit_transform(x_test)
y_pred=model.predict(x_test)

print(model.score(x_train,y_train))
print(model.score(x_test,y_test))
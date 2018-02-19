import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


dataset = pd.read_csv('Salary_Data.csv')
X  =dataset.iloc[:, :-1].values
Y =dataset.iloc[:, 1].values


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3, random_state=0)


#fitting simple linear regression to the training set!
from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(X_train, Y_train)

#prediction
Y_Predict = linearRegression.predict(X_test)

#Plotting for train values
plt.scatter(X_train, Y_train, edgecolors='red' )
plt.plot(X_train,linearRegression.predict(X_train) )
plt.title('Salary v/s Exp')
plt.xlabel('exp years')
plt.ylabel('salary in $')
plt.show()

#Plotting for test values
plt.scatter(X_test, Y_test, edgecolors='red' )
plt.plot(X_train,linearRegression.predict(X_train) )
plt.title('Salary v/s Exp')
plt.xlabel('exp years')
plt.ylabel('salary in $')
plt.show()
#Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
X  =dataset.iloc[:, 3:13].values
Y =dataset.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features= [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#making an ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

classifiernueralnetwork = Sequential()

#adding a  first NN layer & first hidden layer, rectifier - hidden and sigmoid - output layer
#relu = rectfier
classifiernueralnetwork.add(Dense(units = 6, kernel_initializer = "uniform", activation="relu", input_dim=11))

#adding 2nd hidden layer
classifiernueralnetwork.add(Dense(units = 6, kernel_initializer = "uniform", activation="relu"))

#adding output layer
#if dependant variable has 3 or more categories, change op_dim/units to 3 or more categories/classes and
#activation func as softmax
classifiernueralnetwork.add(Dense(units = 1, kernel_initializer = "uniform", activation="sigmoid"))

#compile the ANN - applying stochastic gradient descent on ann - if 3 or more outcome
# then use loss='categorical_crossentropy
classifiernueralnetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics= ["accuracy"])

#fitting ann to the training set
classifiernueralnetwork.fit(X_train,Y_train, batch_size=10 ,nb_epoch=100)

y_pred = classifiernueralnetwork.predict(X_test)
y_pred = (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm =confusion_matrix(Y_test,y_pred)
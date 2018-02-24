import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Data.csv')
X  =dataset.iloc[:, :-1].values
Y =dataset.iloc[:, 3].values

#filling in the missing values by mean, median or most frequent
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy= 'mean', axis=0)
#imputer = Imputer(missing_values= 'NaN', strategy= 'median', axis=0)
#imputer = Imputer(missing_values= 'NaN', strategy= 'most_frequent', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data like texts to number Eg - france as 1, Spain as 2 etc.
#this is done because machine learnig equations deals only with numbers & not text data

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


#Splitting data into traing & test set, use train_test_split class from the sklearn
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#feature scaling to make all the values in one range, like here, age is in tens and
#salary is in 10,000's so there will be issue in computing euclidian distance since it'll
#be dominated by salary only!

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler();
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Scale the dummy variable also for increased accuracy and here I'm not scaling Y since
# it's already in a small range

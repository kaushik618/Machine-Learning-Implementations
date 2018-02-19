import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


dataset = pd.read_csv('50_Startups.csv')
X  =dataset.iloc[:, :-1].values
Y =dataset.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap!

X = X[:, 1:]


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Fitting the linear reg model
from sklearn.linear_model import LinearRegression
multipleRegression = LinearRegression()
multipleRegression.fit(X_train, Y_train)

y_predict = multipleRegression.predict(X_test)

#building an optimal model for backward elimination with a significance level = 0.05
import statsmodels.formula.api as sm
X = np.append( arr=  np.ones((50, 1)).astype(int), values= X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,1,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_ols.summary())


X_opt = X[:, [0,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_ols.summary())

X_opt = X[:, [0,3,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_ols.summary())



X_opt = X[:, [0,3]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_ols.summary())



#Automated Backward elimination algorithm using only p-values
def backwardElimination(X, sl):
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X = np.delete(X, j, 1)
    print(regressor_OLS.summary())
    return X


SL = 0.05
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



#Automated Backward Elimination using R-Squared & p-Values!
def backwardElimination(X, SL):
    numVars = len(X[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = X[:, j]
                    X = np.delete(X, j, 1)
                    tmp_regressor = sm.OLS(Y, X).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((X, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    print(regressor_OLS.summary())
    return X


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


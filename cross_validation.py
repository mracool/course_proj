import logit_reg as lg
from sklearn import cross_decomposition
import parsing as ps
import statsmodels.api as sm
import test as ts
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
import hists as hs
import pandas as pd


general_error = []
column_names = hs.column_names
X_men = ps.data_to_ml[column_names]
X_women = ps.data_to_test[column_names]
temp = 0
columns_with_valid_par = []
columns_with_valid_par.append(column_names[0])


for i in range(1, len(column_names)):
    print(i)
    print(len(columns_with_valid_par))
    error = []
    logreg = LogisticRegression()
    columns_with_valid_par.append(column_names[i])
    logreg.fit(preprocessing.scale(X_men[columns_with_valid_par]), lg.Y_men)

    #  try model on test sample
    y_pred = np.ndarray.tolist(logreg.predict(preprocessing.scale(X_men[columns_with_valid_par])))
    y_test = lg.Y_men.tolist()
    del columns_with_valid_par[-1]
    #  calculating error
    for j in range(len(y_test)):
        if (y_test[j] - y_pred[j]) != 0:
            error.append(1)
        else:
            error.append(0)
    if len(general_error) !=0 and general_error[-1] >= sum(error):
        columns_with_valid_par.append(column_names[i])
    general_error.append(sum(error))
print(y_pred)
print(general_error)

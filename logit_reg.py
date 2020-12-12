# trying pure logreg on raw data

import sklearn
import parsing as ps
import statsmodels.api as sm
import test as ts
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd

data_to_test = ps.data_to_test
data_to_ml = ps.data_to_ml
# columns = ts.columns_to_ml
columns = ts.column_names

X_men = data_to_ml.loc[:, columns]
X_women = data_to_test.loc[:, columns]
Y_men = data_to_ml['Type New']
Y_women = data_to_test['Type New']

Y = pd.concat([Y_men, Y_women])
X = pd.concat([X_women, X_men])
X_train, X_test,  Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.4, random_state=0)

logreg = LogisticRegression()
logreg.fit(preprocessing.scale(X_train), Y_train)


#  try model on test sample
y_pred = np.ndarray.tolist(logreg.predict(preprocessing.scale(X_test)))


#  calculating error
error = 0  #  default is 43%, error with mixed data is 37%
for i in range(len(y_pred)):
    if (y_pred[i] - Y_test.tolist()[i]) != 0:
        error = error + 1
print(error/len(y_pred))

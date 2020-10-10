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
columns = ts.columns_to_ml


X_men = data_to_ml.loc[:, columns]
X_women = data_to_test.loc[:, columns]
Y_men = data_to_ml['Type New']
Y_women = data_to_test['Type New']
# Y = pd.concat([Y_men, Y_women])
# X = pd.concat([X_women, X_men])
# X_train,  Y_train = sklearn.model_selection.train_test_split(X_men, Y_men, test_size=0.4, random_state=0)
# X_test, Y_test = sklearn.model_selection.train_test_split(X_women, Y_women, test_size=0.4, random_state=0)

logreg = LogisticRegression()
logreg.fit(preprocessing.scale(X_men), Y_men)

#  try model on test sample
y_pred = np.ndarray.tolist(logreg.predict(preprocessing.scale(X_women)))

y_test = Y_women.tolist()
print(y_test)
#  calculating error
error = 0
len_y_pred = len(y_pred)
for i in range(len_y_pred):
    if (y_pred[i] - y_test[i]) != 0:
        error = error + 1
print(error/len_y_pred)
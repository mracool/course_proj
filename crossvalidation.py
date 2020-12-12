#crosvalidating selected data to rise up accuracy

from sklearn import preprocessing
import parsing as ps
import test as ts
from sklearn.linear_model import LogisticRegressionCV
import logit_reg as lg
print(lg.Y_women)
data_to_test = ps.data_to_test
data_to_ml = ps.data_to_ml
columns = ts.columns_to_ml


clf = LogisticRegressionCV(cv=3, random_state=1).fit(preprocessing.scale(lg.X_men), lg.Y_men)
clf.predict(preprocessing.scale(lg.X_women)[:2, :])
clf.predict_proba(preprocessing.scale(lg.X_women)[:2, :]).shape

print(clf.score(preprocessing.scale(lg.X_women), lg.Y_women))

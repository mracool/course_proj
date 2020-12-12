# there are realised chi2 test, it didn't work good enough

import pandas as pd
from scipy.stats import chi2_contingency
import parsing as ps
import hists as hs


alpha = 0.1
column_names = hs.column_names
data_to_ml = hs.data_to_ml
# creation example of data to work with
data = ps.data_to_ml
Y = data['Type New']
X = data[column_names]
p_values = []
columns_to_ml = []
f = open("parametrs_to_ml.txt", "w+")
for i in column_names:
    Y = data['Type New']
    X = (data[i])
    data_crosstab = pd.crosstab(X, Y, margins=False)
    rezults = chi2_contingency(data_crosstab)  # вот тут мне кажется метод мутный
    if rezults[1] > alpha:
        columns_to_ml.append(i)
    p_values.append(rezults[1])
    f.write(i+':'+str(rezults))

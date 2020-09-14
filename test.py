import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import chisquare
from pprint import pprint
import parsing as ps
import hists as hs
alpha = 0.1

column_names = hs.column_names
data_to_ml = hs.data_to_ml
# creation example of data to work with
data = ps.data_to_ml
Y = data['Type New']
X = data['B_avg_BODY_att']
p_values = []
columns_to_ml = []
f= open("parametrs_to_ml.txt","w+")
for i in column_names:
    Y = data['Type New']
    X = (data[i])
    data_crosstab = pd.crosstab(X, Y, margins=False)
    rezults= chi2_contingency(data_crosstab) # вот тут мне кажется метод мутный
    if rezults[1] > alpha:
        columns_to_ml.append(i)
    p_values.append(rezults[1])
    f.write(i+':'+str(rezults))
print(p_values)
print(len(columns_to_ml))
print(len(column_names))
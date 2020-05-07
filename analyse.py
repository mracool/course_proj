import parsing
import pprint
from some_boring_calculus import column_names
from some_boring_calculus import how_much
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
data_dict = parsing.data_dict
data = parsing.data
data_mean = {}
data_var = {}

for i in range(0, len(data_dict)):
    if isinstance(data_dict[column_names[i]][2], int) or isinstance(data_dict[column_names[i]][2], float):
        data_mean[column_names[i]] = np.mean(data_dict[column_names[i]])
        # print(data_mean[column_names[i]])
data_type = []
for i in range(0, len(data_dict)):
    data_type.append(type(data_dict[column_names[i]][3]))
    if isinstance(data_dict[column_names[i]][2], int) or isinstance(data_dict[column_names[i]][2], float):
        data_var[column_names[i]] = np.var(data_dict[column_names[i]])
        # print(data_var[column_names[i]])


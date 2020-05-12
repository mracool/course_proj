import pandas as pd
from pprint import pprint
file_name = 'preprocessed_data.csv'
data = pd.read_csv(file_name)
data_dict = {}
for column in data:
    data_dict[column] = list(data[column])
data_to_ml_w1 = {}

#separate data to two sets by sex to test models and to learn models
data_to_test = data[(data["weight_class_Women's Flyweight"] == 1) | (data["weight_class_Women's Featherweight"] == 1) |
                    (data["weight_class_Women's Bantamweight"] == 1) | (data["weight_class_Women's Strawweight"] == 1)]

data_to_ml = data[(data["weight_class_Catch Weight"] == 1)|(data["weight_class_Bantamweight"] == 1)|(data["weight_class_Lightweight"] == 1)| (data["weight_class_Featherweight"] == 1)| (data["weight_class_Flyweight"] == 1)| (data["weight_class_Light Heavyweight"] == 1)| (data["weight_class_Middleweight"] == 1) | (data["weight_class_Open Weight"] == 1) | (data["weight_class_Welterweight"] == 1)| (data["weight_class_Heavyweight"] == 1)]

# 'winner' = 1 if won red | = 0 if blue
iteration = 0
# for index, row in data_to_ml.iterrows():
#     if row['Winner'] == 'Blue':
#         winner = pd.DataFrame({'Winner': 0})
#         data_to_ml.update(winner)
#     else:
#         winner = pd.DataFrame({'Winner': 0})
#         data_to_ml.update(winner)
#     iteration = iteration + 1
Type_new = pd.Series([], dtype='object')
data_to_ml.reset_index(inplace=True)
for i in range(len(data_to_ml)):
    if data_to_ml['Winner'][i] == 'Red':
        Type_new[i] = 1
    elif data_to_ml['Winner'][i] == 'Blue':
        Type_new[i] = 0

data_to_ml.insert(155, 'Type New', Type_new)
Type_new_ts = pd.Series([], dtype='object')
data_to_test.reset_index(inplace=True)
for i in range(len(data_to_test)):
    if data_to_test['Winner'][i] == 'Red':
        Type_new_ts[i] = 1
    elif data_to_test['Winner'][i] == 'Blue':
        Type_new_ts[i] = 0


#   inserting new column with values of list made above
data_to_test.insert(0, 'Type New', Type_new_ts)

data_to_test.pop('Winner')

#data_to_ml.pop('Winner')




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

Y_men = data_to_ml['Type New']
print(columns)
X_men = data_to_ml.loc[:, ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_REV', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_losses', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed', 'B_total_time_fought(seconds)', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Submission', 'B_wins', 'R_draw', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_landed', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_win_by_Decision_Majority', 'R_win_by_TKO_Doctor_Stoppage', 'B_age', 'weight_class_Flyweight', 'weight_class_Lightweight', 'weight_class_Middleweight', 'weight_class_Welterweight', "weight_class_Women's Bantamweight", "weight_class_Women's Featherweight", "weight_class_Women's Flyweight", "weight_class_Women's Strawweight", 'B_Stance_Open Stance', 'B_Stance_Sideways', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Open Stance', 'R_Stance_Switch']
]
X_women = data_to_test.loc[:, ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_REV', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_losses', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed', 'B_total_time_fought(seconds)', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Submission', 'B_wins', 'R_draw', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_landed', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_win_by_Decision_Majority', 'R_win_by_TKO_Doctor_Stoppage', 'B_age', 'weight_class_Flyweight', 'weight_class_Lightweight', 'weight_class_Middleweight', 'weight_class_Welterweight', "weight_class_Women's Bantamweight", "weight_class_Women's Featherweight", "weight_class_Women's Flyweight", "weight_class_Women's Strawweight", 'B_Stance_Open Stance', 'B_Stance_Sideways', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Open Stance', 'R_Stance_Switch']
]
X = pd.concat([X_women,X_men])
Y_women = data_to_test['Type New']
Y = pd.concat([Y_men, Y_women])
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.4, random_state=0)
print(Y)
X_train = preprocessing.scale(X_train)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
#try model on test sample
y_pred = np.ndarray.tolist(logreg.predict(preprocessing.scale(X_test)))
print(y_pred)
Y_test = Y_test.tolist()
err = [0 for x in range(len(Y_test))]
print(type(y_pred),type(Y_test))
error = 0
for i in range(len(y_pred)):
    if (y_pred[i] - Y_test[i]) != 0:
        error = error + 1
print(error/len(y_pred))

#trying pyspark crossvalidation




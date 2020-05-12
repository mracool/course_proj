from pyspark import SparkContext
from pyspark.python.pyspark.shell import sqlContext
from pyspark.shell import spark
import parsing as ps
import test as ts
from pprint import pprint
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import sklearn
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import pandas as pd
from pyspark.sql import SQLContext
from pyspark.sql.types import *


# Auxiliar functions
def equivalent_type(f):
    if f == 'datetime64[ns]':
        return DateType()
    elif f == 'int64':
        return LongType()
    elif f == 'int32':
        return IntegerType()
    elif f == 'float64':
        return FloatType()
    else:
        return StringType()

def define_structure(string, format_type):
    try:
        typo = equivalent_type(format_type)
    except:
        typo = StringType()
    return StructField(string, typo)


# Given pandas dataframe, it will return a spark's dataframe.
def pandas_to_spark(pandas_df):
    columns = list(pandas_df.columns)
    types = list(pandas_df.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
      struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(pandas_df, p_schema)


data_to_test = ps.data_to_test
data_to_ml = ps.data_to_ml
columns = ts.columns_to_ml

X_train = []
X_test = []
Y_test = []
Y = data_to_ml['Type New']
print(columns)
X = data_to_ml.loc[:, ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_REV', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_losses', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed', 'B_total_time_fought(seconds)', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Submission', 'B_wins', 'R_draw', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_landed', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_win_by_Decision_Majority', 'R_win_by_TKO_Doctor_Stoppage', 'B_age', 'weight_class_Flyweight', 'weight_class_Lightweight', 'weight_class_Middleweight', 'weight_class_Welterweight', "weight_class_Women's Bantamweight", "weight_class_Women's Featherweight", "weight_class_Women's Flyweight", "weight_class_Women's Strawweight", 'B_Stance_Open Stance', 'B_Stance_Sideways', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Open Stance', 'R_Stance_Switch']
]
# X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
X_test = data_to_test.loc[:, ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_REV', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_losses', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed', 'B_total_time_fought(seconds)', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Submission', 'B_wins', 'R_draw', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_landed', 'R_avg_SIG_STR_landed', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_longest_win_streak', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_win_by_Decision_Majority', 'R_win_by_TKO_Doctor_Stoppage', 'B_age', 'weight_class_Flyweight', 'weight_class_Lightweight', 'weight_class_Middleweight', 'weight_class_Welterweight', "weight_class_Women's Bantamweight", "weight_class_Women's Featherweight", "weight_class_Women's Flyweight", "weight_class_Women's Strawweight", 'B_Stance_Open Stance', 'B_Stance_Sideways', 'B_Stance_Southpaw', 'B_Stance_Switch', 'R_Stance_Open Stance', 'R_Stance_Switch']
]
Y_test = data_to_test['Type New']
print(Y_test)
Y_train = preprocessing.scale(Y)
X_train = preprocessing.scale(X)
logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
#try model on test sample
# y_pred = np.ndarray.tolist(logreg.predict(preprocessing.scale(X_test)))
# print(y_pred)
# # Y_test = Y_test.tolist()
# err = [0 for x in range(len(Y_test))]
# print(len(y_pred),len(Y_test))
# error = 0
# for i in range(len(y_pred)):
#     if (y_pred[i] - Y_test[i]) != 0 :
#         error = error + 1
# print(error/len(y_pred))

#trying pyspark crossvalidation

training = pandas_to_spark(X)

print("What i've got:")
pprint(training)

crossval = CrossValidator(estimator=logreg, estimatorParamMaps=[], evaluator=BinaryClassificationEvaluator)
cvModel = crossval.fit(training)
# from pyspark.ml import Pipeline
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
# from pyspark.ml.feature import HashingTF, Tokenizer
# from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# from pyspark.python.pyspark.shell import spark
#
# training = spark.createDataFrame([
#     (0, "a b c d e spark", 1.0),
#     (1, "b d", 0.0),
#     (2, "spark f g h", 1.0),
#     (3, "hadoop mapreduce", 0.0),
#     (4, "b spark who", 1.0),
#     (5, "g d a y", 0.0),
#     (6, "spark fly", 1.0),
#     (7, "was mapreduce", 0.0),
#     (8, "e spark program", 1.0),
#     (9, "a e c l", 0.0),
#     (10, "spark compile", 1.0),
#     (11, "hadoop software", 0.0)
# ], ["id", "text", "label"])
#
# # Configure an ML pipeline, which consists of tree stages: tokenizer, hashingTF, and lr.
# tokenizer = Tokenizer(inputCol="text", outputCol="words")
# hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
# lr = LogisticRegression(maxIter=10)
# pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
#
# # We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
# # This will allow us to jointly choose parameters for all Pipeline stages.
# # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
# # We use a ParamGridBuilder to construct a grid of parameters to search over.
# # With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
# # this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
# paramGrid = ParamGridBuilder() \
#     .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
#     .addGrid(lr.regParam, [0.1, 0.01]) \
#     .build()
#
# crossval = CrossValidator(estimator=pipeline,
#                           estimatorParamMaps=paramGrid,
#                           evaluator=BinaryClassificationEvaluator(),
#                           numFolds=2)  # use 3+ folds in practice
#
# # Run cross-validation, and choose the best set of parameters.
# # cvModel = crossval.fit(training)
# #
# # # Prepare test documents, which are unlabeled.
# # test = spark.createDataFrame([
# #     (4, "spark i j k"),
# #     (5, "l m n"),
# #     (6, "mapreduce spark"),
# #     (7, "apache hadoop")
# # ], ["id", "text"])
# #
# # # Make predictions on test documents. cvModel uses the best model found (lrModel).
# # prediction = cvModel.transform(test)
# # selected = prediction.select("id", "text", "probability", "prediction")
# # for row in selected.collect():
# #     print(row)
# print(type(paramGrid))
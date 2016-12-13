# 6th December 2016

import time
from DataSet import DataSet
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix
import visualizations

# Variables
data = DataSet.load_data_set('Split Data_Standard_12-09-2016')
data_type = 'auto'
y_label = 'Normal/Abnormal' # 'Noise'

''' Logistic Regression with Auto Data'''
# Data Preparation
test = data.testing[data_type]
train = data.training[data_type]
val = data.validation[data_type]

test_y = test[[y_label]].values.astype(float).ravel()
# print('Test\n')
# DataSet.print_balance(test_y)
train_y = train[[y_label]].values.astype(float).ravel()
# print('Train\n')
# DataSet.print_balance(train_y)
val_y = val[[y_label]].values.astype(float).ravel()
# print('Val\n')
# DataSet.print_balance(val_y)

test_x = test[data.feature_names[data_type]].values.astype(float)
train_x = train[data.feature_names[data_type]].values.astype(float)
val_x = val[data.feature_names[data_type]].values.astype(float)

# Balance DataSets
test_y,test_x = DataSet.balance_dataset_by_reproduction(test_y,test_x)
train_y,train_x = DataSet.balance_dataset_by_reproduction(train_y,train_x)
val_y,val_x = DataSet.balance_dataset_by_reproduction(val_y,val_x)

# print('Train\n')
# DataSet.print_balance(train_y)

# Create Model
logreg = linear_model.LogisticRegression(penalty='l1')

# Train Model
# Train Parameters with Parameter grid method
# param_grid = {'penalty':['l1','l2'],'C':[1e5,1e4,1e3,1e2,1e1,1,1e-1,1e-2,1e-3]}
param_grid = {'penalty':['l1'],'C':[1]}

best_score = 0
for g in ParameterGrid(param_grid):
    logreg.set_params(**g)
    # weights= DataSet.balance_dataset_by_weights(train_y)
    logreg.fit(train_x, train_y)#,sample_weight=weights)
    # score = logreg.score(val_x,val_y,sample_weight=DataSet.balance_dataset_by_weights(val_y))

    conf = confusion_matrix(val_y, logreg.predict(val_x))
    Sensitivity = conf[0,0]/(conf[0,0]+conf[1,0])
    Specificity = conf[1,1]/(conf[1,1]+conf[0,1])
    score = (Sensitivity+Specificity)/2
    # print(confusion_matrix(train_y, logreg.predict(train_x)))
    if score > best_score:
        best_score = score
        best_grid = g
logreg.set_params(**best_grid)
logreg.fit(train_x, train_y)#,sample_weight=DataSet.balance_dataset_by_weights(train_y))
test_score = logreg.score(test_x,test_y)#,sample_weight=DataSet.balance_dataset_by_weights(test_y))
test_pred_y = logreg.predict(test_x)

# print('Best Score Validation Score: ', best_score)
print('Testing Score: ', test_score)
print('Grid: ', best_grid)

conf = confusion_matrix(test_y, logreg.predict(test_x))
print(conf)
print('Sensitivity: ', conf[0,0]/(conf[0,0]+conf[1,0]))
print('Specificity: ',conf[1,1]/(conf[1,1]+conf[0,1]))




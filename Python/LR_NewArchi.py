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

''' Data Path '''
data = DataSet.load_data_set('Split Data_Standard_12-09-2016')

''' First Separate between Noisy and not Noisy'''

# Variables
data_type = 'auto'
y_label = 'Noise'

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
logreg_noise = linear_model.LogisticRegression(penalty='l1')

# Train Parameters with Parameter grid method
param_grid = {'penalty':['l1'],'C':[1e2,1e1,1,1e-1,1e-2]}

best_score = 0
for g in ParameterGrid(param_grid):
    logreg_noise.set_params(**g)
    # weights= DataSet.balance_dataset_by_weights(train_y)
    logreg_noise.fit(train_x, train_y)#,sample_weight=weights)
    # score = logreg.score(val_x,val_y,sample_weight=DataSet.balance_dataset_by_weights(val_y))

    conf = confusion_matrix(val_y, logreg_noise.predict(val_x))
    Sensitivity = conf[0,0]/(conf[0,0]+conf[1,0])
    Specificity = conf[1,1]/(conf[1,1]+conf[0,1])
    score = (Sensitivity+Specificity)/2
    # print(confusion_matrix(train_y, logreg.predict(train_x)))
    if score > best_score:
        best_score = score
        best_grid = g
logreg_noise.set_params(**best_grid)
logreg_noise.fit(train_x, train_y)#,sample_weight=DataSet.balance_dataset_by_weights(train_y))
test_score = logreg_noise.score(test_x,test_y)#,sample_weight=DataSet.balance_dataset_by_weights(test_y))

# print('Best Score Validation Score: ', best_score)
print('Separation between Noisy and Not Noisy with auto')
print('Testing Score: ', test_score)
print('Grid: ', best_grid)

conf = confusion_matrix(test_y, logreg_noise.predict(test_x))
print(conf)
print('Sensitivity: ', conf[0,0]/(conf[0,0]+conf[1,0]))
print('Specificity: ',conf[1,1]/(conf[1,1]+conf[0,1]))
print('________________________________')

''' Now separate between Abnormal and Normal '''

# Variables
data_type = 'man'
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
logreg_sound = linear_model.LogisticRegression(penalty='l1')

# Train Parameters with Parameter grid method
param_grid = {'penalty':['l1'],'C':[1e2,1e1,1,1e-1,1e-2]}

best_score = 0
for g in ParameterGrid(param_grid):
    logreg_sound.set_params(**g)
    # weights= DataSet.balance_dataset_by_weights(train_y)
    logreg_sound.fit(train_x, train_y)#,sample_weight=weights)
    # score = logreg.score(val_x,val_y,sample_weight=DataSet.balance_dataset_by_weights(val_y))

    conf = confusion_matrix(val_y, logreg_sound.predict(val_x))
    Sensitivity = conf[0,0]/(conf[0,0]+conf[1,0])
    Specificity = conf[1,1]/(conf[1,1]+conf[0,1])
    score = (Sensitivity+Specificity)/2
    # print(confusion_matrix(train_y, logreg.predict(train_x)))
    if score > best_score:
        best_score = score
        best_grid = g
logreg_sound.set_params(**best_grid)
logreg_sound.fit(train_x, train_y)#,sample_weight=DataSet.balance_dataset_by_weights(train_y))
test_score = logreg_sound.score(test_x,test_y)#,sample_weight=DataSet.balance_dataset_by_weights(test_y))

# print('Best Score Validation Score: ', best_score)
print('Separation between Normal and Abnormal with Manual')
print('Testing Score: ', test_score)
print('Grid: ', best_grid)

conf = confusion_matrix(test_y, logreg_sound.predict(test_x))
print(conf)
print('Sensitivity: ', conf[0,0]/(conf[0,0]+conf[1,0]))
print('Specificity: ',conf[1,1]/(conf[1,1]+conf[0,1]))
print('________________________________')


''' Now do final computation '''
# NB: Need to balance data again
test = data.testing['auto']
test_x = test[data.feature_names['auto']].values.astype(float)
test_noise_y = test[['Noise']].values.astype(float).ravel()
test_sound_y = test[['Normal/Abnormal']].values.astype(float).ravel()

predicted_noise_y = logreg_noise.predict(test_x)
predicted_sound_y = logreg_sound.predict(test_x)

matrix = np.zeros((predicted_noise_y.size, 5))
matrix[:, 0] = test_sound_y
matrix[:, 1] = test_noise_y

counter = 0
for i in predicted_noise_y:
    if i == 1:
        if predicted_sound_y[counter] == 1:
            matrix[counter, 2:5] = [1, 0, 0]
        else:
            matrix[counter, 2:5] = [0, 0, 1]
    else:
        matrix[counter, 2:5] = [0, 1, 0]

    counter = counter + 1

Sensitivity, Specificty, MAcc = DataSet.heart_sound_scoring(matrix)

print('Final Results:')
print('Sensitivity: ', Sensitivity)
print('Specificity: ', Specificty)
print('MAcc: ', MAcc)
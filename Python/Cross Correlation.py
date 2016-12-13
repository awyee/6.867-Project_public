# 6th December 2016

from DataSet import DataSet
import numpy as np
import matplotlib.pyplot as plt

# Variables
data = DataSet.load_data_set('Split Data_Standard-&-Specto_12-09-2016')
data_type = 'auto'
y_label = ['Normal/Abnormal','Noise'] # 'Noise'

''' Logistic Regression with Auto Data'''
# Data Preparation
test = data.testing[data_type]
train = data.training[data_type]
val = data.validation[data_type]

test_x = test[data.feature_names[data_type]].values.astype(float)
train_x = train[data.feature_names[data_type]].values.astype(float)
val_x = val[data.feature_names[data_type]].values.astype(float)

data = np.concatenate((test_x,train_x,val_x),axis=0)
cc = np.cov(data.T)

np.savetxt("cc.csv", cc, delimiter=",")



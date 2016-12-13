from DataSet import DataSet
import pickle

# Variables
file_name = 'Split Data_Standard-&-Specto_12-09-2016'
#data=pickle.load(open(file_name+'.p', 'rb'))
data = DataSet.load_data_set(file_name)
data_type = 'auto'
y_label = ['Normal/Abnormal','Noise'] # 'Noise'

''' Logistic Regression with Auto Data'''
# Data Preparation
test = data.testing[data_type]
train = data.training[data_type]
val = data.validation[data_type]

test_y = test[y_label].values.astype(float)#.ravel()
# print('Test\n')
# DataSet.print_balance(test_y)
train_y = train[y_label].values.astype(float)#.ravel()
# print('Train\n')
# DataSet.print_balance(train_y)
val_y = val[y_label].values.astype(float)#.ravel()
# print('Val\n')
# DataSet.print_balance(val_y)

test_x = test[data.feature_names[data_type]].values.astype(float)
train_x = train[data.feature_names[data_type]].values.astype(float)
val_x = val[data.feature_names[data_type]].values.astype(float)

# Balance DataSets
test_y,test_x = DataSet.balance_dataset_by_reproduction(test_y,test_x)
test_y_sound = test_y[:, 0].ravel()
test_y_noise = test_y[:, 1].ravel()
train_y,train_x = DataSet.balance_dataset_by_reproduction(train_y,train_x)
train_y_sound = train_y[:, 0].ravel()
train_y_noise = train_y[:, 1].ravel()
val_y,val_x = DataSet.balance_dataset_by_reproduction(val_y,val_x)
val_y_sound = val_y[:, 0].ravel()
val_y_noise = val_y[:, 1].ravel()

pickle.dump([test_y_sound, test_y_noise, test_x, train_y_sound,train_y_noise,train_x, val_y_sound,val_y_noise,val_x], open(file_name+'_auto_Normal_Abnormal.p', 'wb'), protocol=2)


# data = DataSet.load_data_set(file_name)
# data_type = 'auto'
# y_label = 'Normal/Abnormal' # 'Noise'
#
# file_name = file_name + '_' + data_type + '_' + 'Normal_Abnormal'
#
# ''' Logistic Regression with Auto Data'''
# # Data Preparation
# test = data.testing[data_type]
# train = data.training[data_type]
# val = data.validation[data_type]
#
# test_y = test[[y_label]].values.astype(float).ravel()
# # print('Test\n')
# # DataSetNN.print_balance(test_y)
# train_y = train[[y_label]].values.astype(float).ravel()
# # print('Train\n')
# # DataSetNN.print_balance(train_y)
# val_y = val[[y_label]].values.astype(float).ravel()
# # print('Val\n')
# # DataSet.print_balance(val_y)
#
# test_x = test[data.feature_names[data_type]].values.astype(float)
# train_x = train[data.feature_names[data_type]].values.astype(float)
# val_x = val[data.feature_names[data_type]].values.astype(float)
#
# # Balance DataSets
# test_y,test_x = DataSet.balance_dataset_by_reproduction(test_y,test_x)
# train_y,train_x = DataSet.balance_dataset_by_reproduction(train_y,train_x)
# val_y,val_x = DataSet.balance_dataset_by_reproduction(val_y,val_x)

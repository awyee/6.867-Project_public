# 6th December 2016

from DataSet import DataSet
from sklearn import linear_model, datasets
from sklearn.feature_selection import SelectFromModel

''' Prepare Data '''

# Variables
data = DataSet.load_data_set('Split Data_Standard-&-Specto_12-09-2016')
data_type = 'auto'
y_label = ['Normal/Abnormal','Noise'] # 'Noise'

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

''' Create Model '''
logreg = linear_model.LogisticRegression(penalty='l1',C=1)
logreg.fit(train_x, train_y_sound)
model = SelectFromModel(logreg, prefit=True)

X_new = model.transform(train_x)

print('Old Features: ')
print(train_x.shape)
print('New Features: ')
print(X_new.shape)



# 6th December 2016

import time
from DataSet import DataSet
import os

''' Declare Variables '''
# Fixed Paths
DATA_PATH = '/Users/Nic/Dropbox (MIT)/6.867/6.867 Project/Data'
CURRENT_PATH = os.getcwd()
CONSOLIDATED_LABELS_PATH = '/Consolidated_Labels.xlsx'
DATASET_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

# Variable Paths
FEATURE_FILE_MAN = ['Standard_Feature_data']
FEATURE_FILE_AUTO = ['Standard_Feature_data_auto']
FEATURE_FILE_MAN = DataSet.merge_multiple_feature_files(DATA_PATH,DATASET_LABELS,FEATURE_FILE_MAN)
FEATURE_FILE_AUTO = DataSet.merge_multiple_feature_files(DATA_PATH,DATASET_LABELS,FEATURE_FILE_AUTO)
FEATURE_FILES = [FEATURE_FILE_MAN,FEATURE_FILE_AUTO] # Can't be more than two! and Man before Auto!

date = time.strftime("%m-%d-%Y")

# File Name
file_name = 'Standard'
FILE_NAME = 'Split Data_' + file_name + '_' + date

# Variables
TESTING_FRAC = 0.1
TRAINING_FRAC = 0.8
NORMALISE_FEATURES = True

''' Create Data Set '''
x = DataSet(FILE_NAME, CURRENT_PATH, DATA_PATH, CONSOLIDATED_LABELS_PATH, DATASET_LABELS, TESTING_FRAC,
             TRAINING_FRAC, FEATURE_FILES, NORMALISE_FEATURES)

''' Load Data Set '''
# y = DataSet.load_data_set(FILE_NAME)
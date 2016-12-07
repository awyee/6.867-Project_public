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
FEATURE_FILES = ['Standard_Feature_data']
date = time.strftime("%m-%d-%Y")

# File Name
file_name = 'Split Data_'
for files in FEATURE_FILES:
    file_name += files + '-&-'
file_name = file_name[:-3]
FILE_NAME = file_name + '_' + date

# Variables
TESTING_FRAC = 0.1
TRAINING_FRAC = 0.8
NORMALISE_FEATURES = True

''' Create Data Set '''
x = DataSet(FILE_NAME, CURRENT_PATH, DATA_PATH, CONSOLIDATED_LABELS_PATH, DATASET_LABELS, TESTING_FRAC,
            TRAINING_FRAC, FEATURE_FILES, NORMALISE_FEATURES)

# ''' Load Data Set '''
# y = DataSet.load_data_set(FILE_NAME)
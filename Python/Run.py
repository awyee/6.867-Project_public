# 6th December 2016
# Final Project 6.867

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys
import xlrd
import pandas as pd
import time
from collections import OrderedDict
from DataSet import DataSet
import os

''' Declare Variables '''
# Fixed Paths
DATA_PATH = '/Users/Nic/Dropbox (MIT)/6.867/6.867 Project/Data'
CURRENT_PATH = os.getcwd()
CONSOLIDATED_LABELS_PATH = '/Consolidated_Labels.xlsx'
DATASET_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

# Variable Paths
FEATURE_FILE = 'Standard_Feature_data'
date = time.strftime("%m-%d-%Y")
FILE_NAME = 'Split Data' + '_' + FEATURE_FILE + '_' + date

# Variables
TESTING_FRAC = 0.1
TRAINING_FRAC = 0.8
NORMALISE_FEATURES = True

''' Create Data Set '''
x = DataSet(FILE_NAME, CURRENT_PATH, DATA_PATH, CONSOLIDATED_LABELS_PATH, DATASET_LABELS, TESTING_FRAC,
            TRAINING_FRAC, FEATURE_FILE, NORMALISE_FEATURES)

''' Load Data Set '''
# y = DataSet.load_data_set(FILE_NAME)
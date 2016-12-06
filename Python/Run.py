# 6th December 2016
# Final Project 6.867

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys
import xlrd
import pandas as pd
from collections import OrderedDict
from DataSet import DataSet
import os

# File Paths
DATA_PATH = '/Users/Nic/Dropbox (MIT)/6.867/6.867 Project/Data'
CURRENT_PATH = os.getcwd()
CONSOLIDATED_LABELS_PATH = '/Consolidated_Labels.xlsx'
DATASET_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

# RUN
name = 'Testing_12_06_16'
x = DataSet(name, CURRENT_PATH, DATA_PATH, CONSOLIDATED_LABELS_PATH, DATASET_LABELS, testing_frac=0.1,training_frac=0.8,
            feature_file_name='Standard_Feature_data', normalise_features=True)

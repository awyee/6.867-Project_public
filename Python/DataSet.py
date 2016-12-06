# 5th December
# Data Set Manipulation

from matplotlib import pyplot as plt
import numpy as np
import pickle
import sys
import xlrd
import pandas as pd
from collections import OrderedDict


# def consolidate_feature_files(general_path, general_file_name, dataset_labels, y_values):
#
#     frames = []
#
#     for label in dataset_labels:
#         path = general_path + '/' + label + '/' + general_file_name + '_' + label + '.csv'
#         temp = pd.read_csv(path, skipinitialspace=True, skiprows=0)
#         frames.append(pd.DataFrame(temp))
#
#     out = pd.concat(frames)
#     out.to_csv()
#
#     return pd.concat(frames)


class DataSet:
    """ This class creates the dataset separated into Testing, Training and Validation

        - Data = Testing + (Train + Validation)
            - e.g. Data= 0.1 + (0.8 +0.2)

     """

    def __init__(self, name, current_path, data_path, consolidated_labels_path, data_set_labels, testing_frac=0.1,
                 training_frac=0.80, feature_file_name= 'Standard_Feature_data', normalise_features=True):

        # Names
        self.name = name
        self.data_set_labels = data_set_labels

        # Paths
        self.path = data_path
        self.feature_file_name = feature_file_name
        self.consolidated_labels_path = self.path + consolidated_labels_path
        self.current_path = current_path

        # Data Splitting
        self.testing_frac = testing_frac
        self.training_frac = training_frac

        # Split Data Storage
        self.testing = pd.DataFrame
        self.validation = pd.DataFrame
        self.training = pd.DataFrame

        # Feature Manipulation
        self.normalise = normalise_features
        self.number_features = int()
        self.feature_names = []
        # self.features = pd.DataFrame
        # self.normalised_features = pd.DataFrame

        ''' Extract y-data '''
        # Extract y-Data from each sheet in the worksheet
            # workbook = xlrd.open_workbook(self.consolidated_labels_path)
        # Now create a dictionary for the sheets within the workbook
        ylabel_sheets = dict()
        for labels in self.data_set_labels:
            ylabel_sheets[labels] = pd.read_excel(self.consolidated_labels_path,labels)
            # ylabel_sheets[labels] = workbook.sheet_by_name(labels)

        ''' Extract the features for each Data Set '''
        features_sheets = dict()
        for labels in self.data_set_labels:
            path = self.path + '/' + labels + '/' + self.feature_file_name + '_' + labels + '.csv'
            features_sheets[labels] = pd.read_csv(path, skipinitialspace=True, skiprows=0)

        # Measure the number of features and extract the names
        self.number_features = features_sheets[data_set_labels[1]].shape[1]-1
        self.feature_names = features_sheets[data_set_labels[1]].columns.values

        ''' Combine features and y-data (still a list per each Data Set) '''
        sheets = dict()
        for labels in self.data_set_labels:
            temp = []
            temp.append(ylabel_sheets[labels])
            temp.append(features_sheets[labels])
            sheets[labels] = pd.concat(temp, axis=1, join='outer', join_axes=None, ignore_index=False,
                  keys=None, levels=None, names=None, verify_integrity=False, copy=True)

        ''' Remove zeros if necessary '''
        # for labels in self.data_set_labels:
        for labels in self.data_set_labels:
            temp = sheets[labels]
            sheets[labels] = temp.loc[temp[self.feature_names[1]] != 0]


        ''' Normalise features (careful not to normalise y-data) '''


        ''' Now split the data within each Data Set and finally combine '''
        frames_test = []
        frames_train = []
        frames_validate = []

        for sheet in sheets:
            t_test, t_train, t_validate = self.split_to_ttv(sheets[sheet])
            frames_test.append(t_test)
            frames_train.append(t_train)
            frames_validate.append(t_validate)

        testing = pd.concat(frames_test)
        testing.columns = sheets[data_set_labels[1]].columns

        training = pd.concat(frames_train)
        training.columns = sheets[data_set_labels[1]].columns

        validation = pd.concat(frames_validate)
        validation.columns = sheets[data_set_labels[1]].columns


        ''' Save Data in three separate .csv files '''


        ''' Save the data at the end '''
        self.save_data_set()


    def save_data_set(self):

        pickle.dump(self, open(self.name, 'wb'))  # wb: write and binary

    def split_to_ttv(self, sheet):

        x = pd.DataFrame(sheet.values)
        test = x.sample(frac=self.testing_frac)

        rest = x.drop(test.index)

        train = rest.sample(frac=self.training_frac)
        validate = rest.drop(train.index)

        return test, train, validate

    # def extract_features(self, general_path, general_file_name, dataset_labels, normalise=True):
    #
    #     features = consolidate_feature_files(general_path, general_file_name, dataset_labels)
    #
    #     if(normalise):
    #         # normalised_features = self.NormaliseFeatures
    #         features = features
    #
    #     self.features = features



def load_data_set(filename='dataset.p'):
    '''
      This loads the split data labels
    '''
    return pickle.load(open(filename, 'rb')) # wb: read and binary



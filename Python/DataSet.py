# 5th December
# Data Set Manipulation


import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing


class DataSet:
    """ This class creates the dataset separated into Testing, Training and Validation

        - Data = Testing + (Train + Validation)
            - e.g. Data= 0.1 + (0.8 +0.2)

     """

    def __init__(self, name, current_path, data_path, consolidated_labels_path, data_set_labels, testing_frac=0.1,
                 training_frac=0.80, feature_file_names=None, normalise_features=True):

        # Names
        self.name = name
        self.data_set_labels = data_set_labels

        # Paths
        self.path = data_path
        #self.feature_file_names = feature_file_names
        self.consolidated_labels_path = self.path + consolidated_labels_path
        self.current_path = current_path

        # Data
        self.testing_frac = testing_frac
        self.training_frac = training_frac
        self.features_sheets = feature_file_names  # New
        self.complete_features_sheets = dict()  # Merged together

        # Split Data Storage
        # self.testing = pd.DataFrame
        # self.validation = pd.DataFrame
        # self.training = pd.DataFrame
        #
        # self.testingN = pd.DataFrame
        # self.validationN = pd.DataFrame
        # self.trainingN = pd.DataFrame

        self.testing = dict()
        self.validation = dict()
        self.training = dict()

        # Feature Manipulation
        self.normalise = normalise_features
        self.number_features = int()
        self.feature_names = dict()
        self.feature_label_names = []
        # self.features = pd.DataFrame
        # self.normalised_features = pd.DataFrame

        ''' Extract y-data '''
        # Extract y-Data from each sheet in the worksheet
        # Now create a dictionary for the sheets within the workbook
        ylabel_sheets = dict()
        for labels in self.data_set_labels:
            ylabel_sheets[labels] = pd.read_excel(self.consolidated_labels_path,labels)
        self.feature_label_names = ylabel_sheets[data_set_labels[1]].columns.values

        # ''' Extract the features for each Data Set '''
        # features_sheets = dict()
        # if len(self.feature_file_names) == 1:
        #     feature_file_name = feature_file_names[0]
        #     for labels in self.data_set_labels:
        #         path = self.path + '/' + labels + '/' + feature_file_name + '_' + labels + '.csv'
        #         features_sheets[labels] = pd.read_csv(path, skipinitialspace=True, skiprows=0)
        # else:
        #     features_sheets = self.merge_features()

        # Add a 's_' to feature names of the second file
        for labels in self.data_set_labels:
            self.features_sheets[1][labels].rename(columns=lambda x: 's_'+x, inplace=True)

        # Measure the number of features and extract the names
        self.number_features = self.features_sheets[0][data_set_labels[1]].shape[1] - 1
        self.feature_names['man'] = self.features_sheets[0][data_set_labels[1]].columns.values[1:]
        self.feature_names['auto'] = self.features_sheets[1][data_set_labels[1]].columns.values[1:]
        self.feature_names['both'] = np.append(self.feature_names['man'],self.feature_names['auto'])

        ''' Merge the two feature_sheets '''
        for labels in data_set_labels:
            temp = []
            temp.append(self.features_sheets[0][labels])
            temp2 = self.features_sheets[1][labels]
            del temp2[temp2.columns[0]]
            temp.append(temp2)
            self.complete_features_sheets[labels] = pd.concat(temp, axis=1, join='outer', join_axes=None, ignore_index=False,
                                keys=None, levels=None, names=None, verify_integrity=False, copy=True)

        ''' Combine features and y-data (still a list per each Data Set) '''
        sheets = dict()
        for labels in self.data_set_labels:
            temp = []
            temp.append(ylabel_sheets[labels])
            temp.append(self.complete_features_sheets[labels])
            sheets[labels] = pd.concat(temp, axis=1, join='outer', join_axes=None, ignore_index=False,
                  keys=None, levels=None, names=None, verify_integrity=False, copy=True)

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
        # self.testing = testing

        training = pd.concat(frames_train)
        training.columns = sheets[data_set_labels[1]].columns
        # self.training = training

        validation = pd.concat(frames_validate)
        validation.columns = sheets[data_set_labels[1]].columns
        # self.validation = validation

        ''' Now split between Manual and Auto '''
        man_names = np.append(self.feature_label_names,self.feature_names['man'])
        auto_names = np.append(self.feature_label_names,self.feature_names['auto'])

        self.validation['auto'] = validation.loc[:, auto_names]
        self.validation['man'] = validation.loc[:, man_names]

        self.training['auto'] = training.loc[:, auto_names]
        self.training['man'] = training.loc[:, man_names]

        self.testing['auto'] = testing.loc[:, auto_names]
        self.testing['man'] = testing.loc[:, man_names]

        ''' Remove zeros if necessary '''
        self.validation['man'],self.training['man'],self.testing['man'] = self.remove_zero_rows(self.validation['man'],
                                                                          self.training['man'],self.testing['man'])

        ''' Standardise features (careful not to standardise y-data) '''
        # First we need to take a measure of the size of each t,t,v
        n_val= dict()
        n_train = dict()
        n_test = dict()
        n_val['man'] = self.validation['man'].shape[0]
        n_val['auto'] = self.validation['auto'].shape[0]
        n_train['man']  = self.training['man'] .shape[0]
        n_train['auto'] = self.training['auto'].shape[0]
        n_test['man'] = self.testing['man'].shape[0]
        n_test['auto'] = self.testing['auto'].shape[0]

        all_features = dict()

        temp=[]
        temp.append(self.testing['auto'])
        temp.append(self.validation['auto'])
        temp.append(self.training['auto'])
        all_features['auto'] = pd.concat(temp)
        all_features['auto'].columns = self.testing['auto'].columns

        temp = []
        temp.append(self.testing['man'])
        temp.append(self.validation['man'])
        temp.append(self.training['man'])
        all_features['man'] = pd.concat(temp)
        all_features['man'].columns = self.testing['man'].columns

        # Crashes in the next line when there are NaNs in the data
        std_scale = dict()
        all_features_n = dict()
        std_scale['man'] = preprocessing.StandardScaler().fit(all_features['man'][self.feature_names['man']])
        all_features_n['man'] = std_scale['man'].transform(all_features['man'][self.feature_names['man']])
        std_scale['auto'] = preprocessing.StandardScaler().fit(all_features['auto'][self.feature_names['auto']])
        all_features_n['auto'] = std_scale['auto'].transform(all_features['auto'][self.feature_names['auto']])

        all_features['man'].loc[:, self.feature_names['man']] = all_features_n['man']
        all_features['auto'].loc[:, self.feature_names['auto']] = all_features_n['auto']

        self.testing['man'] = all_features['man'][:n_test['man']-1]
        self.testing['auto'] = all_features['auto'][:n_test['auto'] - 1]

        self.validation['man'] = all_features['man'][:n_val['man'] - 1]
        self.validation['auto'] = all_features['auto'][:n_val['auto'] - 1]

        self.training['man'] = all_features['man'][:n_train['man'] - 1]
        self.training['auto'] = all_features['auto'][:n_train['auto'] - 1]

        ''' Save data so that it may be retrieved '''
        self.save_data_set()

    # def merge_features(self):
    #
    #     features_sheets = dict()
    #     # skip column from second feature
    #     for labels in self.data_set_labels:
    #         temp = []
    #         skip_column = False
    #         for features in self.feature_file_names:
    #             path = self.path + '/' + labels + '/' + features + '_' + labels + '.csv'
    #             sheet = pd.read_csv(path, skipinitialspace=True, skiprows=0)
    #             if skip_column:
    #                 del sheet[sheet.columns[0]]
    #             skip_column = True
    #             temp.append(sheet)
    #         features_sheets[labels] = pd.concat(temp, axis=1, join='outer', join_axes=None, ignore_index=False,
    #                                        keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    #     return features_sheets

    def remove_zero_rows(self, validation, training, testing):

        # N.B. Has to be done before standardization
        # for labels in self.data_set_labels:

        validation = validation.loc[validation[validation.columns[3]] != 0]
        training = training.loc[training[training.columns[3]] != 0]
        testing = testing.loc[testing[testing.columns[3]] != 0]

        return validation, training, testing

    def save_data_set(self):

        pickle.dump(self, open(self.name+'.p', 'wb'))  # wb: write and binary

    def split_to_ttv(self, sheet):
        # Mi sembra funzioni bene ma devo controllare perchè lo mette giallo
        x = pd.DataFrame(sheet.values)
        test = x.sample(frac=self.testing_frac)

        rest = x.drop(test.index)

        train = rest.sample(frac=self.training_frac)
        validate = rest.drop(train.index)

        return test, train, validate

    @staticmethod
    def load_data_set(filename='dataset.p'):
        # This loads the split data labels
        return pickle.load(open(filename+'.p', 'rb'))  # wb: read and binary

    @staticmethod
    def merge_multiple_feature_files(data_path,data_set_labels,feature_file_names):

        features_sheets = dict()
        # skip column from second feature
        for labels in data_set_labels:
            temp = []
            skip_column = False
            for features in feature_file_names:
                path = data_path + '/' + labels + '/' + features + '_' + labels + '.csv'
                sheet = pd.read_csv(path, skipinitialspace=True, skiprows=0)
                if skip_column:
                    del sheet[sheet.columns[0]]
                skip_column = True
                temp.append(sheet)
            features_sheets[labels] = pd.concat(temp, axis=1, join='outer', join_axes=None, ignore_index=False,
                            keys=None, levels=None, names=None, verify_integrity=False, copy=True)
        return features_sheets


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

# def extract_features(self, general_path, general_file_name, dataset_labels, normalise=True):
#
#     features = consolidate_feature_files(general_path, general_file_name, dataset_labels)
#
#     if(normalise):
#         # normalised_features = self.NormaliseFeatures
#         features = features
#
#     self.features = features

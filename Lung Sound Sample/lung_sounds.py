# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:51:46 2016

@author: Daniel Chamberlain
"""

import lung_sound_features
import lung_sound_labels
import settings
import numpy as np
from os.path import join
import classification
from sklearn.cross_validation import train_test_split
import pandas as pd
import lung_sound_deep_learning

def load_labels_and_file_locs(exclude_br=True):
    is_wheeze, is_crackle, is_br, sound_file_locs = lung_sound_labels.load_lung_sound_labels()
    if exclude_br:
        # Drop br
        good_recordings = is_br == 0
        is_wheeze = is_wheeze[good_recordings]
        is_crackle = is_crackle[good_recordings]
        is_br = is_br[good_recordings]
        sound_file_locs = sound_file_locs[good_recordings]

    return is_wheeze,is_crackle,is_br,sound_file_locs

def get_features(sound_file_locs,feature_type='engineered',deep_model_num=[]):
    if feature_type == 'engineered':
        x = generate_engineered_features(sound_file_locs)
    elif feature_type == 'sda' or feature_type == 'conv':
        data = lung_sound_deep_learning.get_spectrograms(norm_type='area',spec_keys=sound_file_locs)
        x = lung_sound_deep_learning.get_features_from_model(data,model_type=feature_type,model_num=deep_model_num)
        x = pd.DataFrame(data=x,index=sound_file_locs)
    else:
        print('Unknown feature type:' + feature_type)
        return
    return x


def generate_engineered_features(sound_file_locs):
    
    # Convert paths for local os
    # Can't use the os version for key because data might be loaded on different platforms
    sound_file_keys = sound_file_locs
    local_os_path = settings.load_lung_sound_path()
    sound_file_locs = np.asarray([join(local_os_path,f) for f in sound_file_locs])    
    
    
    x_sound = lung_sound_features.generate_sound_features(sound_file_locs,sound_file_keys)
    x_spec = lung_sound_features.generate_spectrogram_features(sound_file_locs,sound_file_keys)
    x_wheeze_spec = lung_sound_features.generate_spectrogram_features(sound_file_locs,sound_file_keys,wheeze_spec=True)
    x_patch_5x5 = lung_sound_features.generate_spectrogram_patch_features(sound_file_locs,sound_file_keys,patch_size_x=5,patch_size_y=5,patch_type='small')
    x_patch_5xw = lung_sound_features.generate_spectrogram_patch_features(sound_file_locs,sound_file_keys,patch_size_x=5,patch_size_y=5,patch_type='vertical')
    x_patch_5xh = lung_sound_features.generate_spectrogram_patch_features(sound_file_locs,sound_file_keys,patch_size_x=5,patch_size_y=5,patch_type='horizontal')
    x_spike = lung_sound_features.generate_spike_counter_features(sound_file_locs,sound_file_keys)
    x_wheeze = lung_sound_features.generate_wheeze_detector_features(sound_file_locs,sound_file_keys)
    x = pd.concat([x_sound,x_spec,x_wheeze_spec,x_patch_5x5,x_patch_5xw,
                   x_patch_5xh,x_spike,x_wheeze],axis=1)    
    return x
    

def get_lung_sound_classifiers(feature_type='engineered',
                               deep_model_num = [],
                               clf_type='rf',verbose=0):
    if verbose>0:   
        print('Loading labels')
    is_wheeze,is_crackle,is_br,sound_file_locs = load_labels_and_file_locs()
    if verbose>0:
        print('Generating features')
        
    x_frame = get_features(sound_file_locs,
                     feature_type=feature_type,
                     deep_model_num=deep_model_num)
    x = x_frame.get_values().astype(float)
    # Create wheeze classifier
    y = is_wheeze
    if verbose>0:
        print('Training wheezing classifier')    
    wheeze_clf,has_prob = classification.train_classifier(x,y,clf_type=clf_type)
         
    # Create crackle classifer
    y = is_crackle
    if verbose>0:
        print('Training crackle classifier')    
    crackle_clf,has_prob = classification.train_classifier(x,y,clf_type=clf_type)
    return wheeze_clf,crackle_clf,has_prob
    
    
def test_model(outcome='wheeze',clf_type='rf',feature_type='engineered',deep_model_num=[]):
    print('Load labels')
    if outcome == 'wheeze' or outcome == 'crackle':
        is_wheeze,is_crackle,is_br,sound_file_locs = load_labels_and_file_locs(exclude_br=True)
        if outcome == 'wheeze':
            y = is_wheeze
        else:
            y = is_crackle
    elif outcome == 'br':
        is_wheeze,is_crackle,is_br,sound_file_locs = load_labels_and_file_locs(exclude_br=False)
        y = is_br
        
    print('Generate features')    
    x = get_features(sound_file_locs,
                     feature_type=feature_type,
                     deep_model_num=deep_model_num)
    # Evaluate wheeze classifier
    
    percent_test = 0.2
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     test_size=percent_test,
                                                     stratify=y)
    print('Training classifier')
    clf,has_prob = classification.train_classifier(x_train,y_train,clf_type=clf_type,verbose=0)
    print('Evaluating classifier')
    auc = classification.evaluate_classifier(clf,x_test,y_test,has_prob=has_prob)
#    print('Wheeze test set auc: {:0.03f}'.format(wheeze_auc))
##    clf_type = 'rf'
#    print('Train wheeze classifier')    
#    clf,has_prob = classification.train_classifier(x_train,y_train,clf_type=clf_type,verbose=0)
#    print('Evaluate model')
#    wheeze_auc = classification.evaluate_classifier(clf,x_test,y_test,has_prob=has_prob)
#    print('Wheeze test set auc: {:0.03f}'.format(wheeze_auc))
##        
#    # Evaluate crackle classifer
#    y = is_crackle
#    percent_test = 0.2    
#    x_train,x_test,y_train,y_test = train_test_split(x,y,
#                                                     test_size=percent_test,
#                                                     stratify=y)
#    
#    print('Train crackle classifier')    
#    clf,has_prob = classification.train_classifier(x_train,y_train,clf_type=clf_type)
#    print('Evaluate model')
#    crackle_auc = classification.evaluate_classifier(clf,x_test,y_test,has_prob=has_prob)
#    print('Crackle test set auc: {:0.03f}'.format(crackle_auc))
    return auc
    
    
if __name__ == '__main__':
    num_trials = 5
#    feature_types = ['engineered','sda']
    outcomes = ['wheeze','crackle','br']
    feature_types = ['engineered','sda','conv']
    model_ranges = [[0],np.arange(6),np.arange(8)]
    clf_types = ['rf','svc']
    with open('results/lung_sound_results.csv','w') as f:
        f.write('outcome,feature_type,deep_model_num,clf_type,auc_mean,auc_std\n')
    for outcome in outcomes:
        for feature_type_num in np.arange(len(feature_types)):
            feature_type = feature_types[feature_type_num]
            deep_model_nums = model_ranges[feature_type_num]
            for deep_model_num in deep_model_nums:
                for clf_type in clf_types:
                    aucs = []
                    for trial in np.arange(num_trials):
                        print(trial)
                        auc = test_model(feature_type=feature_type,deep_model_num=deep_model_num,clf_type=clf_type)
                        aucs.append(auc)
                    with open('results/lung_sound_results.csv','a') as f:
                        f.write('{},{},{},{},{:0.03f},{:0.03f}\n'.format(outcome,feature_type,deep_model_num,clf_type,np.mean(aucs),np.std(aucs)))

                

#    test_model()
#    test_model(feature_type='engineered')
#    test_model(feature_type='sda',deep_model_num=0)
#    test_model(feature_type='sda',deep_model_num=1)
#    test_model(feature_type='sda',deep_model_num=2)
#    test_model(feature_type='sda',deep_model_num=3)
#    test_model(feature_type='sda',deep_model_num=4)
#    test_model(feature_type='sda',deep_model_num=5)
    
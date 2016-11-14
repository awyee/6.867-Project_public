# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 14:06:28 2016

@author: Daniel Chamberlain
"""


"""
TODO
# TODO Current Symptoms: Extract information other than just presence or absence of symptoms
# TODO Risk Factors: include type of allergies, number of hours per day cooking, years cooking
# TODO Risk Factors: include type of smoking, start date, stop date, number per day
# TODO Extract other symptoms
# TODO Extract vitals
# TODO Extract medical history
# TODO include type of allergies, number of hours per day cooking, years cooking

"""



import pandas
import numpy as np
import sklearn
import sklearn_pandas
from sklearn.cross_validation import train_test_split
import classification
from os.path import join, isfile, isdir
import os
import lung_sounds
import settings
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import itertools

def parse_datasheet(features_to_use,
                    lung_sound_algorithm_feature_type='engineered',
                    lung_sound_algorithm_deep_model_num = [],
                    lung_sound_algorithm_clf_type='rf'):
    # Load in file
    filename = 'data/Steto study_119+47=166 forms.xlsx'
    patient_data = pd.read_excel(filename,sheetname=1,index_col=3)

    # Drop patients with nan in diagnosis
    patient_data = patient_data.loc[patient_data.loc[:,'N10_Diagnosis'].notnull(),:]

    # Extract diagnosis
    has_asthma, has_copd, had_tb, is_healthy = extract_diagnosis(patient_data.loc[:,['N10_Diagnosis','N91_a1_Diagnosis 1','N91_a2_Diagnosis 2','N91_a3_Diagnosis 3']])

    x = pd.DataFrame()
    if 'lung_sound_algorithm' in features_to_use:
        # Extract lung sound features
        x_lung_sounds = extract_algorithm_lung_sound_features(patient_data.index.get_values(),
                                                              feature_type=lung_sound_algorithm_feature_type,
                                                              deep_model_num=lung_sound_algorithm_deep_model_num,
                                                              clf_type=lung_sound_algorithm_clf_type)
        x.join(x_lung_sounds,how='outer') 
    elif 'lung_sound_doctor' in features_to_use:
        x_lung_sounds = extract_doctor_lung_sound_features(patient_data.loc[:,'N13_1aLung sound heard in area':'N13_11bLung sound heard in area'])
        x = x.join(x_lung_sounds,how='outer')     

    if 'pfm' in features_to_use:
        pfm_frame = patient_data.loc[:,'N15_1_Reading 1':'N15_5_Reading 5']
        pfm_population_normal_frame = patient_data.loc[:,['N6_Sex','N7_Age in Years','N8_Height in cm']]
        x_pfm = extract_pfm_readings(pfm_frame,pfm_population_normal_frame)
        x = x.join(x_pfm,how='outer')
    
    if 'questionnaire' in features_to_use:
        # Extract demographics
        x_demographic = extract_demographics(patient_data.loc[:,['N6_Sex','N7_Age in Years','N9_Weight in Kg']])
        # Extract occuptation and socioeconomic status
        x_ses = extract_ses(patient_data.loc[:,'N16_Occupation':'N17_Socioeconomic status'])
        # Extract current symptoms
        x_symptoms = extract_current_symptoms(patient_data.loc[:,'N18_Breathlessness':'N71_Sneezing'])
        # TODO Extract other symptoms
        # Extract risk factors
        x_risk_factors = extract_risk_factors(patient_data.loc[:,'N83_Family history of COPD':'N90_Notes_Riskfactors'])
        # Extract personal history
        x_personal_history = extract_personal_history(patient_data.loc[:,'N103_a_Smoking (Present or Past)':'N109_Any other habbit'])
    # TODO Extract vitals
    # TODO Extract medical history
    # Combine all of the predictors        
        x = x.join([x_demographic,x_ses,x_symptoms,x_risk_factors,
                       x_personal_history],how='outer')           
                   
    return has_asthma, has_copd, had_tb, is_healthy, x


def extract_diagnosis(diagnosis_frame):
    has_asthma = diagnosis_frame.iloc[:,0].str.contains('ASTHMA').values
    has_copd = diagnosis_frame.iloc[:,0].str.contains('COPD').values
    had_tb = diagnosis_frame.iloc[:,0].str.contains('POST TB').values
    is_healthy = diagnosis_frame.iloc[:,0].str.contains('HEALTHY').values
    # How well does this agree with N91?
    # has_asthma_91 = diagnosis_frame.iloc[:,1].str.contains('ASTHMA') & \
    #                 diagnosis_frame.iloc[:,2].str.contains('ASTHMA') & \
    #                 diagnosis_frame.iloc[:,3].str.contains('ASTHMA')
    # has_copd_91 = diagnosis_frame.iloc[:,1].str.contains('COPD') & \
    #                 diagnosis_frame.iloc[:,2].str.contains('COPD') & \
    #                 diagnosis_frame.iloc[:,3].str.contains('COPD')
    # has_asthma == has_asthma_91
    return has_asthma, has_copd, had_tb, is_healthy


def extract_doctor_lung_sound_features(lung_sounds_frame):
    wheeze_flag = np.zeros((len(lung_sounds_frame.index),11))
    crackle_flag = np.zeros((len(lung_sounds_frame.index),11))
    normal_flag = np.zeros((len(lung_sounds_frame.index),11))
    # Get a wheeze and crackle indicator for each location
    for lung_sound_area in np.arange(0,11):
        col_name = 'N13_{}aLung sound heard in area'.format(lung_sound_area+1)
        lung_sounds_area = lung_sounds_frame.loc[:,col_name]

        # Check for wheezes
        wheeze_flag[:,lung_sound_area] = (lung_sounds_area.str.contains('WP') | lung_sounds_area.str.contains('WM')).values
        crackle_flag[:,lung_sound_area] = (lung_sounds_area.str.contains('CF') | lung_sounds_area.str.contains('CC')).values
        normal_flag[:,lung_sound_area] = (lung_sounds_area.str.contains('N')).values

    # Data verification
    if np.min(wheeze_flag + crackle_flag + normal_flag) < 1:
        print('Unknown sound detected at location: ',np.where(wheeze_flag + crackle_flag + normal_flag < 1))
    # Determine if patient has any abnormal breathing sounds
    num_abnormal_sounds = np.sum(wheeze_flag + crackle_flag,axis=1)
    num_wheezes = np.sum(wheeze_flag,axis=1)
    num_wheezes_upper = np.sum(wheeze_flag[:,[0,1,2,9,10]],axis=1)
    num_wheezes_lower = np.sum(wheeze_flag[:,[3,4,5,6,7,8]],axis=1)
    num_crackles = np.sum(crackle_flag,axis=1)
    num_crackles_upper = np.sum(crackle_flag[:,[0,1,2,9,10]],axis=1)
    num_crackles_lower = np.sum(crackle_flag[:,[3,4,5,6,7,8]],axis=1)
    

    x = np.zeros((crackle_flag.shape[0],10))
    x[:,0] = num_abnormal_sounds > 0
    x[:,1] = num_abnormal_sounds
    x[:,2] = num_wheezes > 0
    x[:,3] = num_wheezes
    x[:,4] = num_crackles > 0
    x[:,5] = num_crackles
    x[:,6] = num_wheezes_upper
    x[:,7] = num_wheezes_lower
    x[:,8] = num_crackles_upper
    x[:,9] = num_crackles_lower
    
    feature_names = np.asarray(['abnormal_sound_flag', 'num_abnormal_sounds','wheeze_flag','num_wheezes',
                                'crackle_flag','num_crackles','num_wheezes_upper',
                                'num_wheezes_lower','num_crackles_upper',
                                'num_crackles_lower'])
    x = pd.DataFrame(data=x,index = lung_sounds_frame.index,columns = feature_names)
    return x


def extract_algorithm_lung_sound_features(patient_ids,
                                          feature_type='engineered',
                                          deep_model_num = [],
                                          clf_type = 'rf',
                                          verbose=0):   
    # Generate and save model
    
    if feature_type == 'sda' or feature_type == 'conv':
        clf_file = 'cache/lung_sound_classifiers_F'+feature_type + '_C' + clf_type + '_M{}'.format(deep_model_num) + '.pkl'
    elif feature_type == 'engineered':
        clf_file = 'cache/lung_sound_classifiers_F'+feature_type + '_C' + clf_type + '.pkl'
    else:
        print('Feature type not implemented')
        return
    if not isfile(clf_file):
        wheeze_clf,crackle_clf,has_prob = lung_sounds.get_lung_sound_classifiers(feature_type=feature_type,
                                                                                 deep_model_num = deep_model_num,clf_type=clf_type)
        with open(clf_file, 'wb') as save_file:
            pickle.dump(wheeze_clf,save_file)
            pickle.dump(crackle_clf,save_file)
            pickle.dump(has_prob,save_file)
    else:
        with open(clf_file, 'rb') as save_file:
            wheeze_clf = pickle.load(save_file)
            crackle_clf = pickle.load(save_file)
            has_prob = pickle.load(save_file)
    
    if feature_type == 'sda' or feature_type == 'conv':
        feature_file = 'cache/lung_sounds_algorithm_F'+feature_type + '_C' + clf_type + '_M{}'.format(deep_model_num) + '.pkl'
    elif feature_type == 'engineered':
        feature_file = 'cache/lung_sounds_algorithm_F'+feature_type + '_C' + clf_type + '.pkl'
        
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
        x = pd.DataFrame()
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)    
        
    # Determine lung sound data folder
    study_loc = 'PhaseII_CRF'
    local_os_path = settings.load_lung_sound_path()
    # Iterate through and process sound files
    for m in np.arange(len(patient_ids)):
        if(m % np.floor(len(patient_ids)/100) == 0):
            if verbose > 0:
                print(m,' of ',len(patient_ids))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
                features_changed = False
        diagnosis_patient_id = patient_ids[m]
        sound_patient_id = patient_ids[m]
        # Strip letters from sound patient IDs
        sound_patient_id = str.split(sound_patient_id,'/')[1]
        patient_folder = join(local_os_path,study_loc,sound_patient_id)
        if isdir(patient_folder):
            patient_key = study_loc + '/' + sound_patient_id
            if not diagnosis_patient_id in x.index or np.any(x.loc[diagnosis_patient_id].isnull()) or len(x.columns) == 0:
                features_changed = True
                sound_files = np.asarray(os.listdir(patient_folder))
                for area_num in np.arange(1,12):
                    if verbose > 0:
                        print('Area num ', area_num,' of ',11)
                    matching_file_locs = np.core.defchararray.find(sound_files,'Area {:02d}'.format(area_num))>=0
                    if np.sum(matching_file_locs) < 13:
                        print('Less than 13 sound files for patient {} area {}'.format(patient_key,area_num))
                    matching_sound_files_key = [patient_key + '/' + f for f in sound_files[matching_file_locs]]
                    area_sound_features_frame = lung_sounds.get_features(matching_sound_files_key,feature_type=feature_type,deep_model_num=deep_model_num)                    
                    area_sound_features = area_sound_features_frame.get_values().astype(float)
                    p = classification.predict(wheeze_clf,area_sound_features,has_prob=has_prob)
                    x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMean'.format(area_num)] = np.mean(p)
                    x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMax'.format(area_num)] = np.max(p)
                    p = classification.predict(crackle_clf,area_sound_features,has_prob=has_prob)
                    x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMean'.format(area_num)] = np.mean(p)
                    x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMax'.format(area_num)] = np.max(p)
                    if area_num == 1:
                        patient_wheeze_area_mean = x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMean'.format(area_num)]
                        patient_crackle_area_mean = x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMean'.format(area_num)]
                        patient_wheeze_area_max = x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMax'.format(area_num)]                    
                        patient_crackle_area_max = x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMax'.format(area_num)]
                    else:
                        patient_wheeze_area_mean += x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMean'.format(area_num)]
                        patient_crackle_area_mean =+ x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMean'.format(area_num)]                    
                        patient_wheeze_area_max = max([patient_wheeze_area_max,x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_A{:02d}_ProbMax'.format(area_num)]])
                        patient_crackle_area_max = max([patient_crackle_area_max,x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_A{:02d}_ProbMax'.format(area_num)]])
                x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_ProbMean'] = patient_wheeze_area_mean/11
                x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_ProbMean'] = patient_crackle_area_mean/11
                x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Wheeze_ProbMax'] = patient_wheeze_area_max
                x.loc[diagnosis_patient_id,'LungSoundAlgorithm_Crackle_ProbMax'] = patient_crackle_area_max
        else:
            print('Missing lung sounds from patient ',diagnosis_patient_id)
            x.loc[diagnosis_patient_id] = np.NaN
    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
            
    x_keys = x.loc[patient_ids,['LungSoundAlgorithm_Wheeze_ProbMean','LungSoundAlgorithm_Wheeze_ProbMax','LungSoundAlgorithm_Crackle_ProbMean','LungSoundAlgorithm_Crackle_ProbMax']] 
    return x_keys


def extract_demographics(demographic_frame):
    # Convert sex from 2=female,1=male to 0 = female,1=male
    demographic_frame.loc[demographic_frame['N6_Sex'] == 2,'N6_Sex'] = 0
    demographic_frame.columns = ['sex','age_years','weight_kg']
    demographic_frame = demographic_frame[['sex','weight_kg']] # Exclude age
    return demographic_frame

def extract_pfm_readings(pfm_frame,demographic_frame):
    x = pd.DataFrame(index = pfm_frame.index)
    x['max_pfm_reading'] = pfm_frame.max(axis=1, skipna=True)

    reference_values = ((demographic_frame.loc[:,'N6_Sex'] == 1).values.astype(int) * \
                       (-1.807 * demographic_frame.loc[:,'N7_Age in Years'].values +
                        3.206 * demographic_frame.loc[:,'N8_Height in cm'].values)) + \
                       ((demographic_frame.loc[:,'N6_Sex'] == 2).values.astype(int) *
                        (-1.454*demographic_frame.loc[:,'N7_Age in Years'].values +
                         2.368 * demographic_frame.loc[:,'N8_Height in cm'].values))
    x['max_pfm_reading_over_reference'] = np.divide(x['max_pfm_reading'],reference_values)
    return x


def extract_ses(ses_frame):
    ses_frame['ses_string'] = ses_frame['N17_Socioeconomic status'].map({1:'high',2:'middle',3:'low'})
    x = pd.get_dummies(ses_frame['ses_string'],prefix='ses')
    # TODO implement occupation
#    feature_names = ['ses']
    return x


def extract_current_symptoms(symptom_frame):
    
    # TODO Extract information other than just presence or absence of symptoms
    breathlessness_mapper = sklearn_pandas.DataFrameMapper([
        (['N18_Breathlessness'], [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    cough_mapper = sklearn_pandas.DataFrameMapper([
        (['N30_Cough'], [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    chest_pain_mapper = sklearn_pandas.DataFrameMapper([
        (['N50_Chest Pain'], [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    fever_mapper = sklearn_pandas.DataFrameMapper([
        (['N55_Fever'], [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    nasal_mapper = sklearn_pandas.DataFrameMapper([
        (['N64_Nasal Symptoms'], [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes

    x_breathlessness = breathlessness_mapper.fit_transform(symptom_frame.copy())
    x_cough = cough_mapper.fit_transform(symptom_frame.copy())
    x_chest_pain = chest_pain_mapper.fit_transform(symptom_frame.copy())
    x_fever = fever_mapper.fit_transform(symptom_frame.copy())
    x_nasal = nasal_mapper.fit_transform(symptom_frame.copy())


    x = pd.DataFrame(index=symptom_frame.index)
    x['has_breathlessness'] = x_breathlessness
    x['has_cough'] = x_cough
    x['has_chest_pain'] = x_chest_pain
    x['has_fever'] = x_fever
    x['has_nasal_symptoms'] = x_nasal
    return x


def extract_risk_factors(risk_factor_frame):
    risk_factor_mapper = sklearn_pandas.DataFrameMapper([
        (['N83_Family history of COPD', 'N84_Family hostory of allergies',
          'N86_Personal History of allergies?','N87_Indoor cooking using Biomass?'],
         [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    # TODO include type of allergies, number of hours per day cooking, years cooking
    
    x_risk_factor = risk_factor_mapper.fit_transform(risk_factor_frame.copy())
    feature_names = ['has_copd_family_history','has_allergy_family_history','has_allergy_personal_history',
                     'has_biomass_cooking_history']
    x = pd.DataFrame(data=x_risk_factor,index = risk_factor_frame.index,columns = feature_names)
    return x


def extract_personal_history(personal_history_frame):
    personal_history_mapper = sklearn_pandas.DataFrameMapper([
        (['N103_a_Smoking (Present or Past)', 'N107_Tobacco Chewing',
          'N108_Alcohol Intake'],
         [sklearn.preprocessing.FunctionTransformer(impute2,validate=False), # Impute no
                                sklearn.preprocessing.FunctionTransformer(np.negative),
                                sklearn.preprocessing.Binarizer(threshold = -1.5)])]) # Flip order so 0 is no, 1 is yes
    # TODO include type of smoking, start date, stop date, number per day

    x = personal_history_mapper.fit_transform(personal_history_frame.copy())
    feature_names = ['has_smoked','has_chewed_tobacco','has_drunk_alcohol']
    x = pd.DataFrame(data=x,index = personal_history_frame.index,columns = feature_names)
    return x



def impute2(x):
    x[np.isnan(x)] = 2
    return x


def load_data(outcome,
              exclude_na_behavior='cols',
              features_to_use = ['lung_sound_algorithm','pfm','questionnaire']):
    
    has_asthma, has_copd, had_tb, is_healthy, x = parse_datasheet(features_to_use,
                                                                  lung_sound_algorithm_feature_type='sda',
                                                                  lung_sound_algorithm_deep_model_num=4,
                                                                  lung_sound_algorithm_clf_type='svc')

    if outcome == 'obs_vs_nonobs':
        # y = np.logical_or(has_asthma,np.logical_or(has_copd,had_tb))
        y = np.logical_or(has_asthma,has_copd)
    elif outcome == 'asthma_vs_copd':
        asthma_or_copd = np.logical_or(has_asthma,has_copd)
        x = x.iloc[asthma_or_copd]
        y = has_asthma[asthma_or_copd]
    elif outcome == 'healthy_vs_unhealthy':
        y = is_healthy
    elif outcome == 'multiclass':
        y = np.zeros(has_asthma.shape)
        y += 1*has_asthma
        y += 2*has_copd
        # y += 3*had_tb
    else:
        print('Outcome unknown:',outcome)

    if exclude_na_behavior == 'cols':
        null_cols = x.isnull().any().get_values()
        print('Dropping the following features because of NA values:',x.columns.values[null_cols])
        x = x.dropna(axis=1)
    elif exclude_na_behavior == 'rows':
        null_rows = x.isnull().any(axis=1).get_values()
        print('Excluding the following patients because of NA values:',x.index.values[null_rows])        
        y = y[np.logical_not(null_rows)]
        x = x.iloc[np.logical_not(null_rows)]
    else:
        print('Please specify an NA behavior.')

    
    return x,y


def create_and_analyze_classifier(outcome,
                                  lung_sound_algorithm_flag,
                                  exclude_na_behavior='rows'):
    x_frame,y = load_data(outcome,
                          lung_sound_algorithm_flag = lung_sound_algorithm_flag,
                          exclude_na_behavior = exclude_na_behavior)
    x = x_frame.get_values().astype(float)
    # Split dataset
    percent_test = 0.2
    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     test_size=percent_test,
                                                     stratify=y)      
    
    clf,has_prob = classification.train_classifier(x_train,y_train,clf_type='lr',verbose=1)
    auc = classification.evaluate_classifier(clf,x_test,y_test,has_prob=has_prob)
    print('Test set auc: {:0.03f}'.format(auc))
    coef = np.squeeze(clf.named_steps['clf'].coef_)
    classification.plot_feature_importance(x_frame.columns.values,coef)
#    plt.title('AUC: {:0.03f}'.format(auc))
    plt.tight_layout()
    plt.savefig('img/feature_importance_' + outcome + '_LSA' + str(int(lung_sound_algorithm_flag)) + '.png')
    plt.close()
    
    classification.plot_roc(clf,x_test,y_test,has_prob=has_prob)
    plt.savefig('img/roc_curve' + outcome + '_LSA' + str(int(lung_sound_algorithm_flag)) + '.png')
    plt.close()    
    

def compare_feature_sets(outcome):
    feature_types = np.asarray(['lung_sound_doctor','pfm','questionnaire'])
    feature_combos = np.asarray(list(itertools.product([0,1], repeat=3))).astype(bool)
    feature_combos = feature_combos[1:,:]
    
    output_file = 'results/diagnosis_feature_importance_results.csv'
    if not os.path.isfile(output_file):
        with open(output_file,'w') as f:
            f.write('outcome,lung_sounds,pfm,questionnaire,auc_median,auc_low,auc_high\n')
    for m in np.arange(len(feature_combos)):
        x_frame,y = load_data(outcome,
                              features_to_use=feature_types[feature_combos[m,:]],
                              exclude_na_behavior = 'rows')
        x = x_frame.get_values().astype(float)
        clf,has_prob = classification.train_classifier(x,y,clf_type='lr',verbose=0)
        auc_median,auc_low,auc_high= classification.generate_average_roc(x,y,clf,has_prob)
        plt.savefig('img/feature_importance/roc_' + outcome + '_LS{:d}_PF{:d}_QU{:d}.png'.format(feature_combos[m,0],feature_combos[m,1],feature_combos[m,2]))
        plt.close()
        with open(output_file,'a') as f:
            f.write('{},{:d},{:d},{:d},{:0.03f},{:0.03f},{:0.03f}\n'.format(outcome,feature_combos[m,0],feature_combos[m,1],feature_combos[m,2],auc_median,auc_low,auc_high))
#        coef = np.squeeze(clf.named_steps['clf'].coef_)
#        classification.plot_feature_importance(x_frame.columns.values,coef)
#        plt.savefig('img/feature_importance/feature_importance_' + outcome + '_LS1_LSA1.png')    
#        plt.close()
    
    

    
if __name__ == '__main__':
    compare_feature_sets('asthma_vs_copd')
    compare_feature_sets('obs_vs_nonobs')
    compare_feature_sets('healthy_vs_unhealthy')
#    create_and_analyze_classifier('asthma_vs_copd',True)
#    create_and_analyze_classifier('asthma_vs_copd',False)
#    create_and_analyze_classifier('obs_vs_nonobs',True)
#    create_and_analyze_classifier('obs_vs_nonobs',False)
    
#    create_and_analyze_classifier('asthma_vs_copd',True)
#    create_and_analyze_classifier('asthma_vs_copd',False)
#    create_and_analyze_classifier('obs_vs_nonobs',True)
#    create_and_analyze_classifier('obs_vs_nonobs',False)
    
#    compare_with_and_without_lung_sounds('obs_vs_nonobs')
#    compare_with_and_without_lung_sounds('asthma_vs_copd')
#    compare_with_and_without_lung_sounds('healthy_vs_unhealthy')
                    
                    
                    
                    
#    extract_algorithm_lung_sound_features(np.asarray(['RKJ/11001']))
    
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:19:00 2016

@author: Daniel
"""

import pandas as pd
import numpy as np
#import scipy.io as sio
from os.path import join
#import csv
#import zipfile
#import wave
#import struct
import pickle

def load_lung_sound_labels(sound_file_key_loc = 'data/soundFileKey_2016-04-27.pkl'):
    # Load in sound file labels and the unblinded filenames
    save_file = open(sound_file_key_loc,'rb')
    sound_file_key = pickle.load(save_file)
    save_file.close()

    # Exclude first 110 files becuase there was an error in labeling
    sound_file_key = sound_file_key[110:,:]

#    if not verify_sound_file_key(sound_file_key,local_os_path):
#        return 
        
    # There are now two labeling formats
    # Load original spreadsheet labels
    filename = 'data/SoundFileLabels_2016-04-19.xlsx'
    xl = pd.ExcelFile(filename)
    sheet_names = xl.sheet_names
    m = 0
    for sheet_name in sheet_names:
        data = pd.read_excel(filename, sheetname=sheet_name,header = 14)
        lung_sound_label_strings = data[['First Sound Heard', 'Second Sound Heard', 'Revised label with extended files']].values
        sound_file_names_blinded_sheet = data['Filename'].values
        is_wheeze_sheet = np.zeros((lung_sound_label_strings.shape[0]))
        is_crackle_sheet = np.zeros((lung_sound_label_strings.shape[0]))
        is_bad_recording_sheet = np.zeros((lung_sound_label_strings.shape[0]))
        
        for row_num in range(lung_sound_label_strings.shape[0]):
            if(pd.isnull(lung_sound_label_strings[row_num,2]) and not pd.isnull(lung_sound_label_strings[row_num,1])):
                # We have more than one sound labeled for this sound file
                this_wheeze, this_crackle, this_br = interpret_lung_sound_label(lung_sound_label_strings[row_num,0])
                this_wheeze_2, this_crackle_2, this_br_2 = interpret_lung_sound_label(lung_sound_label_strings[row_num,1])

                if((this_br == 1) or (this_br_2 == 1)):
                    is_bad_recording_sheet[row_num] = 1
                else:
                    if((this_wheeze == 1) or (this_wheeze_2 == 1)):
                        is_wheeze_sheet[row_num] = 1
                    if((this_crackle == 1) or (this_crackle_2 == 1)):
                        is_crackle_sheet[row_num] = 1
            else:
                if(not pd.isnull(lung_sound_label_strings[row_num,2])):
                    # We had a revised label after listening to a longer sound file
                    this_wheeze,this_crackle,this_br = interpret_lung_sound_label(lung_sound_label_strings[row_num,2])
                elif not pd.isnull(lung_sound_label_strings[row_num,0]):
                    # We did not have a revised label and we have a label for this file
                    this_wheeze,this_crackle,this_br = interpret_lung_sound_label(lung_sound_label_strings[row_num,0])
                else:
                    # No label has yet been provided for this sound file
                    this_wheeze = -1
                    this_crackle = -1
                    this_br = -1
                is_wheeze_sheet[row_num] = this_wheeze
                is_crackle_sheet[row_num] = this_crackle
                is_bad_recording_sheet[row_num] = this_br
        if m == 0:
            is_wheeze = is_wheeze_sheet
            is_crackle = is_crackle_sheet
            is_br = is_bad_recording_sheet
            sound_file_names_blinded = sound_file_names_blinded_sheet
            m +=1
        else:
            is_wheeze = np.hstack((is_wheeze,is_wheeze_sheet))
            is_crackle = np.hstack((is_crackle,is_crackle_sheet))
            is_br = np.hstack((is_br,is_bad_recording_sheet))
            sound_file_names_blinded = np.hstack((sound_file_names_blinded,sound_file_names_blinded_sheet))

    ## Drop sound files that have not yet been labeled
    labeled_locs = is_wheeze >= 0
    is_br = is_br[labeled_locs]
    is_crackle = is_crackle[labeled_locs]
    is_wheeze = is_wheeze[labeled_locs]
    sound_file_names_blinded = sound_file_names_blinded[labeled_locs]

    # Load app labels
    filename = 'data/lungSoundLabels_2016-06-20.csv'
    app_labels = pd.read_csv(filename)
    is_wheeze_app, is_crackle_app, is_br_app, need_longer_app, sound_file_names_blinded_app = interpret_app_labels(app_labels)
    
    # Exclude files where we need a longer recording
    is_wheeze_app = is_wheeze_app[np.logical_not(need_longer_app)]
    is_crackle_app = is_crackle_app[np.logical_not(need_longer_app)]
    is_br_app = is_br_app[np.logical_not(need_longer_app)]
    sound_file_names_blinded_app = sound_file_names_blinded_app[np.logical_not(need_longer_app)]
    
    # Combine labels
    is_wheeze = np.hstack((is_wheeze,is_wheeze_app))
    is_crackle = np.hstack((is_crackle,is_crackle_app))
    is_br = np.hstack((is_br,is_br_app))
    sound_file_names_blinded = np.hstack((sound_file_names_blinded,sound_file_names_blinded_app))

    # Unblind filenames
    sound_file_locs_unblinded = unblind_filenames(sound_file_names_blinded,sound_file_key)
    # Remove duplicates
    sound_file_locs_unblinded, indices = np.unique(sound_file_locs_unblinded,return_index=True)
    # sound_file_locs_unblinded = sound_file_locs_unblinded[indices]
    is_wheeze = is_wheeze[indices]
    is_crackle = is_crackle[indices]
    is_br = is_br[indices]

#    # Find spec locs
#    spec_locs = convert_sound_locs_to_spec_locs(sound_file_locs_unblinded)

    sound_file_locs_unblinded = np.core.defchararray.replace(sound_file_locs_unblinded.astype(str),"data/shortSoundFiles/raw/","")
    return is_wheeze, is_crackle, is_br, sound_file_locs_unblinded


def interpret_app_labels(app_labels):
    sound_file_names_blinded_app = app_labels['soundFileName'].values   
#    is_br_app = np.logical_or(app_labels['isBr'].values,app_labels['needLonger'].values)
    is_br_app = app_labels['isBr'].values
    is_wheeze_app = app_labels['isWheeze'].values
    is_crackle_app = app_labels['isCrackle'].values
    need_longer_app = app_labels['needLonger'].values
    return is_wheeze_app, is_crackle_app, is_br_app, need_longer_app, sound_file_names_blinded_app


def interpret_lung_sound_label(lung_sound_label):
    if((lung_sound_label == 'WP') or (lung_sound_label == 'WM')):
        return 1,0,0
    elif((lung_sound_label == 'CC') or (lung_sound_label == 'CF')):
        return 0,1,0
    elif(lung_sound_label == 'NM'):
        return 0,0,0
    elif(lung_sound_label == 'BR'):
        return 0,0,1
    elif(lung_sound_label == 'Cough'):
        return 0,0,1
    elif(lung_sound_label == 'SQ'):
        return 0,0,1
    else:
        print(lung_sound_label + ' not a recognized lung sound label')

def unblind_filenames(blinded_names,key):
    key_blind = key[:,0]
    key_unblind = key[:,1]

#    # Remove empty unblinded strings
#    empty_flag = np.zeros((len(key_unblind)))
#    m = 0
#    for this_key_unblind in key_unblind:
#        if not key_unblind[m]:
#            empty_flag[m] = 1
#        m+=1
#    # key_blind = key_blind[empty_flag != 1]
#    # key_unblind = key_unblind[empty_flag != 1]

    # Strip path from key_blind
    m = 0
    for this_key_blind in key_blind:
        backslash_loc = np.core.defchararray.rfind(this_key_blind,'/')
        if backslash_loc >= 0:
            key_blind[m] = this_key_blind[backslash_loc+1:len(this_key_blind)]
        m+=1
    blinded_names = np.core.defchararray.replace(blinded_names.astype(str),"'","")

    m = 0
    for blinded_name in blinded_names:
        match_loc = np.where(key_blind==blinded_name)[0]
        if len(match_loc) > 0:
            item_index = np.where(key_blind==blinded_name)[0][0]
            unblinded_name = key_unblind[item_index]
            if m == 0:
                unblinded_names = np.asarray(unblinded_name)
            else:
                unblinded_names = np.hstack((unblinded_names,unblinded_name))
            m+=1
#            if empty_flag[m]:
#                unblinded_names = np.hstack((unblinded_names,['']))
#        else:
#            print('Could not find a match for',blinded_name)
    return unblinded_names


#def convert_sound_locs_to_spec_locs(sound_locs):
#    sound_file_path = 'data/shortSoundFiles/raw/'
#    spec_file_path = 'data/STFT025_norm_area/'
#    spec_locs = np.core.defchararray.add(np.core.defchararray.replace(sound_locs,sound_file_path,spec_file_path),'.mat')
#    return spec_locs


def verify_sound_file_key(sound_file_key,local_os_path):
    is_matching = np.zeros(np.size(sound_file_key,0))
    for m in np.arange(np.size(sound_file_key,0)):
        if np.size(sound_file_key[m,0]) > 0:
            sound_file_blind, fs_blind = load_sound_file(join(local_os_path,sound_file_key[m,0]))
            if np.size(sound_file_key[m,1]) > 0:    
                sound_file_unblind, fs_unblind = load_sound_file(join(local_os_path,sound_file_key[m,1]))                
                is_matching[m] = sound_file_blind == sound_file_unblind
    if np.sum(is_matching) == len(is_matching):
        return True
    else:
        print('Sound file entries are not equal')
        return False
            
        
#def load_sound_file(sound_file_loc):
#    wave_file = wave.open(sound_file_loc,'rb')
#    fs = wave_file._framerate
#    sound_data = wave_file.readframes(wave_file.getnframes())
#    sound_data = struct.unpack("%ih" % (wave_file.getnframes()* wave_file.getnchannels()), sound_data)
#    return sound_data, fs


#def package_teaching_app_sound_files():
#    is_wheeze, is_crackle, is_br, sound_file_locs_unblinded, spec_loc = load_lung_sound_labels()
#    # Exclude br
#    good_file = np.logical_not(is_br)
#    is_wheeze = is_wheeze[good_file]
#    is_crackle = is_crackle[good_file]
#    is_br = is_br[good_file]
#    sound_file_locs_unblinded = sound_file_locs_unblinded[good_file]
#    is_normal = np.logical_not(np.logical_or(is_wheeze,np.logical_or(is_crackle,is_br))).astype(int)
#
#    # Exclude br
#
#    # Generate output sound file names
#    original_path = \
#        '/home/daniel/Dropbox (MIT)/Research/pulmonary-diagnostics/lung-sound-identification/data/shortSoundFiles/raw/'
#    sound_file_output_name = np.core.defchararray.lower(np.core.defchararray.replace(np.core.defchararray.replace(
#        np.core.defchararray.replace(sound_file_locs_unblinded,original_path,''),'/','_'),' ','_'))
#
#    # Write .csv file
#    label_file = 'lung_sound_labels.csv'
#    with open(label_file, 'w', newline='') as csvfile:
#        writer = csv.writer(csvfile, delimiter='|',
#                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
#        header_row = ['sound_file'] + ['is_br'] + ['is_crackle'] + ['is_wheeze'] + ['is_pleural_rub'] + ['is_normal']
#        writer.writerow(header_row)
#        for m in np.arange(len(is_wheeze)):
#            row = [sound_file_output_name[m]] + [is_br[m]] +  [is_crackle[m]] + [is_wheeze[m]] + [0] + [is_normal[m]]
#            writer.writerow(row)
#
#    with zipfile.ZipFile('teaching_app_sound_files.zip', 'w') as myzip:
#        myzip.write(label_file)
#        for m in np.arange(len(is_wheeze)):
#            myzip.write(sound_file_locs_unblinded[m],arcname=sound_file_output_name[m])


if __name__ == '__main__':
    print('Nothing here yet')
#    package_teaching_app_sound_files()
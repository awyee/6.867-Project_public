# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:15:30 2016

@author: Daniel
"""
import os
from os.path import join,isdir

def remove_csv_lung_sounds():
    lung_sound_path = 'data/LungSoundsPhaseII'
    patient_folders = os.listdir(lung_sound_path)
    patient_folders = [join(lung_sound_path,f) for f in patient_folders if isdir(join(lung_sound_path,f))]
    for patient_folder in patient_folders:
        files = os.listdir(patient_folder)
        csv_files = [join(patient_folder,f) for f in files if '.csv' in f]
        for csv_file in csv_files:        
            os.remove(csv_file)
            
            
            
if __name__ == '__main__':
    remove_csv_lung_sounds()
        
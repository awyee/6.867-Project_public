# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 06:47:26 2016

@author: Daniel
"""

import settings
from os.path import join, isfile, isdir
from os import listdir
import lung_sound_features
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import autoencoder
import conv_autoencoder

def get_sound_file_list():
    # Generate complete list of sound files
    sound_files = []
    sound_file_keys = []
    local_os_path = settings.load_lung_sound_path()
    sound_file_locs = 'data/shortSoundFiles/raw'
    lung_sound_path = join(local_os_path,sound_file_locs)
    loc_folders = listdir(lung_sound_path)
    for loc_folder in loc_folders:
        loc_path = join(lung_sound_path,loc_folder)
        if isdir(loc_path):
            patient_folders = listdir(loc_path)
            for patient_folder in patient_folders:
                patient_path = join(loc_path,patient_folder)
                if isdir(patient_path):
                    patient_sound_files = listdir(patient_path)
                    patient_file_keys = [join(loc_folder,patient_folder,f) for f in patient_sound_files]
                    sound_file_keys = sound_file_keys + patient_file_keys
                    patient_sound_files = [join(patient_path,f) for f in patient_sound_files]
                    sound_files = sound_files + patient_sound_files
    
    sorted_order = np.argsort(sound_file_keys)   
    sound_files = np.asarray(sound_files)[sorted_order]
    sound_file_keys = np.asarray(sound_file_keys)[sorted_order]
    
    return sound_files,sound_file_keys    
    
    
def get_spectrograms(im_scale=25,norm_type='all',norm_range=100,spec_keys=[]):
    if norm_type != 'area' and norm_type != 'all':
        print('Unknown norm type')
        return
    
    output_file = 'cache/lung-sound-deep-learning/spec_IS{:03d}_NT{}_NR{:03d}.pkl'.format(im_scale,norm_type,norm_range)
    if not isfile(output_file):
        sound_files,sound_file_keys = get_sound_file_list()
        # Determine spec size
        sound_data,fs = lung_sound_features.load_sound(sound_files[0])
        p,f,t = lung_sound_features.compute_spectrogram(sound_data,fs)
        p = scipy.misc.imresize(p,im_scale,interp='bicubic',mode='F')
        spec_size = p.shape        
        
        # Loop through and generate all spectrograms to create normalization    
        raw_spec_file = 'cache/lung-sound-deep-learning/raw_specs_IS{:03d}.pkl'.format(im_scale)
        if not isfile(raw_spec_file):
            specs = np.zeros((len(sound_files),spec_size[0],spec_size[1]))
            for m in np.arange(len(sound_files)):
                if(m % np.floor(len(sound_files)/10) == 0):
                    print(m,' of ',len(sound_files))
                sound_data,fs = lung_sound_features.load_sound(sound_files[m])
                p,f,t = lung_sound_features.compute_spectrogram(sound_data,fs)
                p = scipy.misc.imresize(p,im_scale,interp='bicubic',mode='F')
                if np.sum(np.isnan(p))>0:
                    print('Problem at m = {}'.format(m))
                    break
                specs[m,:,:] = p
            with open(raw_spec_file, 'wb') as save_file:
                pickle.dump(specs,save_file)
                pickle.dump(sound_file_keys,save_file)
                pickle.dump(f,save_file)
                pickle.dump(t,save_file)
        else:
            with open(raw_spec_file, 'rb') as save_file:
                specs = pickle.load(save_file)
                sound_file_keys = pickle.load(save_file)
                f = pickle.load(save_file)
                t = pickle.load(save_file)
        if norm_type == 'all':
            freq_mean = np.mean(specs,axis=(0,2),keepdims=True)
            freq_std = np.std(specs,axis=(0,2),keepdims=True)
            specs_norm = np.divide(specs - freq_mean,freq_std)
        elif norm_type == 'area':
            specs_norm = np.zeros(specs.shape)
            freq_mean = np.zeros((11,spec_size[0]))
            freq_std = np.zeros((11,spec_size[0]))
            for area_num in np.arange(1,12):
                area_files = np.core.defchararray.find(sound_file_keys,'Area {0:02d}'.format(area_num)) > 0
                specs_area = specs[area_files,:,:]
                area_freq_mean = np.mean(specs_area,axis=(0,2),keepdims=True)
                area_freq_std = np.std(specs_area,axis=(0,2),keepdims=True)
                specs_norm[area_files,:,:] = np.divide(specs_area - area_freq_mean,area_freq_std)
                freq_mean[area_num-1,:] = np.squeeze(area_freq_mean)
                freq_std[area_num-1,:] = np.squeeze(area_freq_std)
        with open(output_file, 'wb') as save_file:
            pickle.dump(specs_norm,save_file)
            pickle.dump(sound_file_keys,save_file)
            pickle.dump(freq_mean,save_file)
            pickle.dump(freq_std,save_file)
            pickle.dump(f,save_file)
            pickle.dump(t,save_file)
    else:
        with open(output_file, 'rb') as save_file:
            specs_norm = pickle.load(save_file)
            sound_file_keys = pickle.load(save_file)
            
    if len(spec_keys) > 0:
        specs_norm_small = np.zeros((len(spec_keys),specs_norm.shape[1],specs_norm.shape[2]))
        for m in np.arange(len(spec_keys)):
            matching_loc = spec_keys[m] == np.asarray(sound_file_keys)
            specs_norm_small[m,:,:] = specs_norm[matching_loc,:,:]
        specs_norm = specs_norm_small
            
    return specs_norm
    
    
def create_model(model_num,model_type='sda',data_norm_type='area',overwrite=False,layer_sizes = [50,50,50],
                                      corruption_levels=[0.3,0.3,0.3],learning_rate=0.01):
    specs = get_spectrograms(norm_type=data_norm_type)
    
    output_folder = 'lung-sound-deep-learning-models/{}_{}'.format(model_type,model_num)
    if isdir(output_folder) and not overwrite: # Throw error is model already exists
        print('Model already exists:' + output_folder)
        return

    if model_type == 'sda':
        specs = np.expand_dims(specs,1)
        autoencoder.train_autoencoder(specs,output_folder,batch_size=20,learning_rate=learning_rate,num_epochs=10000,
                                      momentum_flag=True,momentum=0.9,layer_sizes = layer_sizes,
                                      corruption_levels=corruption_levels,nonlinearity='sigmoid')
    elif model_type == 'conv':
        specs = np.expand_dims(specs,1)
        conv_autoencoder.train_conv_autoencoder(specs,output_folder,(20,49),learning_rate=learning_rate)                              
    else:
        print('Model type not implemented:' + model_type)
        return


def get_features_from_model(data,model_num,model_type='sda',data_norm_type='area'):
    output_folder = 'lung-sound-deep-learning-models/{}_{}'.format(model_type,model_num)
    if model_type == 'sda':
        data = np.expand_dims(data,1)
        l_encode = autoencoder.load_model(output_folder)
        x = autoencoder.generate_features_from_model(data,l_encode)
    elif model_type == 'conv':
        data = np.expand_dims(data,1)
        l_encode = conv_autoencoder.load_model(output_folder)
        x = conv_autoencoder.generate_features_from_model(data,l_encode)
    else:
        print('Model type not implemented:' + model_type)
        return    
    return x


if __name__ == '__main__':
    print('Here')
#    a = get_spectrograms(norm_type='area')
#    create_model(0,overwrite=True,layer_sizes=[10,10,10],corruption_levels=[0.3,0.3,0.3])
#    create_model(1,overwrite=True,layer_sizes=[50,50,50],corruption_levels=[0.3,0.3,0.3])
#    create_model(2,overwrite=True,layer_sizes=[250,250,250],corruption_levels=[0.3,0.3,0.3])
#    create_model(3,overwrite=True,layer_sizes=[10,10,10],corruption_levels=[0.1,0.2,0.3])
#    create_model(4,overwrite=True,layer_sizes=[50,50,50],corruption_levels=[0.1,0.2,0.3])
#    create_model(5,overwrite=True,layer_sizes=[250,250,250],corruption_levels=[0.1,0.2,0.3])
    create_model(6,model_type='conv',learning_rate=0.001)
    create_model(7,model_type='conv',learning_rate=0.0001)
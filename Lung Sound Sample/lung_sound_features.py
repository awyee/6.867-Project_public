# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:13:45 2016

@author: Daniel
"""


import scipy.io as sio
import scipy.signal
from matplotlib import pyplot as plt
# import bob.ip.gabor.Wavelet
import numpy as np
import scipy
import scipy.stats
from os.path import isfile
import pickle
import wave
import struct
import sklearn.feature_extraction.image
import os
import pandas as pd


def load_sound(sound_file_loc):
    wave_file = wave.open(sound_file_loc,'rb')
    fs = wave_file.getframerate()
    sound_data = wave_file.readframes(-1)
    sound_data = np.fromstring(sound_data,'Int16')
    return sound_data,fs
    
def compute_spectrogram(sound_data,fs,win_time=0.04,exclude_dc = True):
    # Create spectrogram
    win_length = win_time*fs
    [f,t,s] = scipy.signal.spectrogram(sound_data,fs=fs,window=scipy.signal.get_window('hamming', win_length),nperseg=win_length,noverlap = win_length//2)
    s[s==0] = 10**-7
    p = np.log10(np.abs(s))
    if exclude_dc:
        # Exclude DC offset        
        p = p[1:,:] 
        f = f[1:]
    return p,f,t


def generate_sound_features(sound_file_locs,sound_file_keys,verbose=0):
    feature_file = 'cache/sound_features.pkl'
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
#        var_names = ['lf_argmax','mf_argmax','hf_argmax','lf_max_over_lf_mean',
#                     'mf_max_over_mf_mean','hf_max_over_hf_mean',
#                     'lf_mean_over_mf_mean','lf_mean_over_hf_mean',
#                     'mf_mean_over_hf_mean','kurtosis','renyi_entropy',
#                     'zero_crossing_std_dev',
#                     'zero_crossing_std_dev_over_zero_crossing_mean',
#                     'f50_over_f90']
        x = pd.DataFrame(index=sound_file_keys)
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)
    
    
    for m in np.arange(len(sound_file_locs)):
        if(m % np.floor(len(sound_file_locs)/10) == 0):
            if verbose > 0:
                print(m,' of ',len(sound_file_locs))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
        sound_file_loc = sound_file_locs[m]
        sound_file_key = sound_file_keys[m]
        if not sound_file_key in x.index or np.any(x.loc[sound_file_key].isnull()) or len(x.columns) == 0:
            features_changed = True
            sound_data,fs = load_sound(sound_file_loc)
    
            # Compute fft
            fft_result = np.abs(np.fft.fft(sound_data))
            freq = np.fft.fftfreq(len(sound_data),1/fs)
            fft_result = fft_result[0:int(len(fft_result)/2)]
            freq = freq[0:int(len(freq)/2)]
    
            ## frequencyAnalysisCrackles.m
            # Get frequency bins
            lf = np.logical_and(freq >= 100, freq < 250)
            mf = np.logical_and(freq >= 250, freq < 800)
            hf = np.logical_and(freq >= 800, freq < 2000)
    
            # Extract freq with max value in each range
            x.loc[sound_file_key,'lf_argmax'] = freq[lf][np.argmax(fft_result[lf])]
            x.loc[sound_file_key,'mf_argmax'] = freq[mf][np.argmax(fft_result[mf])]
            x.loc[sound_file_key,'hf_argmax'] = freq[hf][np.argmax(fft_result[hf])]
    
            # Get peak strength in each section normalized by average strength
            x.loc[sound_file_key,'lf_max_over_lf_mean'] = np.max(fft_result[lf]) / np.mean(fft_result[lf])
            x.loc[sound_file_key,'mf_max_over_mf_mean'] = np.max(fft_result[mf]) / np.mean(fft_result[mf])
            x.loc[sound_file_key,'hf_max_over_hf_mean'] = np.max(fft_result[hf]) / np.mean(fft_result[hf])
    
            # Get ratios between powers in different sections
            x.loc[sound_file_key,'lf_mean_over_mf_mean'] = np.mean(fft_result[lf]) / np.mean(fft_result[mf])
            x.loc[sound_file_key,'lf_mean_over_hf_mean'] = np.mean(fft_result[lf]) / np.mean(fft_result[hf])
            x.loc[sound_file_key,'mf_mean_over_hf_mean'] = np.mean(fft_result[mf]) / np.mean(fft_result[hf])
    
            ## kurtosisDetector.m
            x.loc[sound_file_key,'kurtosis'] = scipy.stats.kurtosis(sound_data) # Kurtosis
            x.loc[sound_file_key,'renyi_entropy'] = renyi_entropy(sound_data) # Entropy
            crossing_irregularity, crossing_irregularity_norm = zero_crossing_irregularity(sound_data) # Mean crossing
            x.loc[sound_file_key,'zero_crossing_std_dev'] = crossing_irregularity
            x.loc[sound_file_key,'zero_crossing_std_dev_over_zero_crossing_mean'] = crossing_irregularity_norm
            # f50/f90
            fft_cumsum = np.cumsum(fft_result)/np.sum(fft_result)
            f50 = freq[np.where(fft_cumsum > 0.5)[0][0]]
            f90 = freq[np.where(fft_cumsum > 0.9)[0][0]]
            x.loc[sound_file_key,'f50_over_f90'] = f50/f90 # f50/f90
    

    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
#            config.write(configfile)
#        sio.savemat(feature_file,{'x':x,'feature_names':feature_names})
            
    x_keys = x.loc[sound_file_keys]
    return x_keys


def renyi_entropy(signal,alpha = 2,bins = 20):
    hist,binedges = np.histogram(signal, range=(-32768,32767), bins=bins)
    p = hist/len(signal)
    entropy = 1/(1-alpha) * np.log2(np.sum(np.power(p, alpha)))
    return entropy


def zero_crossing_irregularity(signal):
    signal = signal - np.mean(signal)
    t1=signal[0:len(signal)-1]
    t2=signal[1:len(signal)]
    tt=np.multiply(t1,t2)
    idx = np.where(tt < 0)
    mean_crossing_distance = np.diff(idx)
    crossing_irregularity = np.std(mean_crossing_distance)
    crossing_irregularity_norm = crossing_irregularity/np.mean(mean_crossing_distance)
    return crossing_irregularity, crossing_irregularity_norm


def generate_spectrogram_features(sound_file_locs,sound_file_keys,
                                  wheeze_spec = False,win_time = 0.04,
                                  verbose = 0):
    if wheeze_spec:
        feature_file = 'cache/wheeze_spec_features.pkl'
        feature_prefix = 'wheeze_'
        wheeze_min_freq = 250
        wheeze_max_freq = 1000
    else:
        feature_file = 'cache/spec_features.pkl'
        feature_prefix = ''        
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
        x = pd.DataFrame(index=sound_file_keys)
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)    
    
    for m in np.arange(len(sound_file_locs)):
        if(m % np.floor(len(sound_file_locs)/10) == 0):
            if verbose > 0:
                print(m,' of ',len(sound_file_locs))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
        sound_file_loc = sound_file_locs[m]
        sound_file_key = sound_file_keys[m]
        if not sound_file_key in x.index or np.any(x.loc[sound_file_key].isnull()) or len(x.columns) == 0:
            features_changed = True
            sound_data,fs = load_sound(sound_file_loc)
    
                
            p,f,t = compute_spectrogram(sound_data,fs,win_time=win_time)
            if wheeze_spec:
                wheeze_freq_bins = (f >= wheeze_min_freq) & (f <= wheeze_max_freq)
                p = p[wheeze_freq_bins,:]
                
    
            # Mean features
            x.loc[sound_file_key,feature_prefix+'spec_mean_power'] = np.mean(p.flatten()) # Mean overall value
            x.loc[sound_file_key,feature_prefix+'spec_max_mean_power_by_time'] = np.max(np.mean(p,axis=0)) # Max of mean at each time
            x.loc[sound_file_key,feature_prefix+'spec_max_mean_power_by_freq'] = np.max(np.mean(p,axis=1)) # Max of mean at each freq
            x.loc[sound_file_key,feature_prefix+'spec_var_mean_power_by_time'] = np.var(np.mean(p,axis=0)) # Var of mean at each time
            x.loc[sound_file_key,feature_prefix+'spec_var_mean_power_by_freq'] = np.var(np.mean(p,axis=1)) # Var of mean at each freq
            # Variance features
            x.loc[sound_file_key,feature_prefix+'spec_var_power'] = np.var(p.flatten()) 
            x.loc[sound_file_key,feature_prefix+'spec_max_var_power_by_time'] = np.max(np.var(p,axis=0))
            x.loc[sound_file_key,feature_prefix+'spec_max_var_power_by_freq'] = np.max(np.var(p,axis=1))
            x.loc[sound_file_key,feature_prefix+'spec_var_var_power_by_time'] = np.var(np.var(p,axis=0))
            x.loc[sound_file_key,feature_prefix+'spec_var_var_power_by_freq'] = np.var(np.var(p,axis=1))
            x.loc[sound_file_key,feature_prefix+'spec_mean_var_power_by_time'] = np.mean(np.var(p,axis=0))
            x.loc[sound_file_key,feature_prefix+'spec_mean_var_power_by_freq'] = np.mean(np.var(p,axis=1))
            # Max locations
            x.loc[sound_file_key,feature_prefix+'spec_argmax_mean_power_by_freq'] = f[np.argmax(np.mean(p,axis=1))]
            x.loc[sound_file_key,feature_prefix+'spec_argmax_var_power_by_freq'] = f[np.argmax(np.var(p,axis=1))]
            x.loc[sound_file_key,feature_prefix+'spec_argmax_var_over_mean_power_by_freq'] = f[np.argmax(np.var(p,axis=1)/np.mean(p,axis=1))]

    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
            
    x_keys = x.loc[sound_file_keys]
    return x_keys


def generate_spectrogram_patch_features(sound_file_locs,sound_file_keys,
                                        win_time = 0.04,patch_size_x = 5,
                                        patch_size_y = 5,patch_type = 'small',
                                        verbose = 0):           
    if patch_type == 'small':
        feature_prefix = 'patch_small_w{}_h{}'.format(patch_size_x,patch_size_y)  
    elif patch_type == 'vertical':
        feature_prefix = 'patch_vertical_w{}'.format(patch_size_x)
    elif patch_type == 'horizontal':
        feature_prefix = 'patch_horizontal_h{}'.format(patch_size_y)
    else:
        print('Unknown patch type:' + patch_type)
        return -1
    feature_file = 'cache/'+feature_prefix+'.pkl'
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
        x = pd.DataFrame(index=sound_file_keys)
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)    
    
    for m in np.arange(len(sound_file_locs)):
        if(m % np.floor(len(sound_file_locs)/10) == 0):
            if verbose > 0:
                print(m,' of ',len(sound_file_locs))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
        sound_file_loc = sound_file_locs[m]
        sound_file_key = sound_file_keys[m]
        if not sound_file_key in x.index or np.any(x.loc[sound_file_key].isnull()) or len(x.columns) == 0:
            features_changed = True
            sound_data,fs = load_sound(sound_file_loc)
    
            # Create spectrogram
            p,f,t = compute_spectrogram(sound_data,fs,win_time=win_time) 
    
           
            if patch_type == 'small':
                patches = sklearn.feature_extraction.image.extract_patches_2d(p, (patch_size_x,patch_size_y))
            elif patch_type == 'vertical':
                patches = sklearn.feature_extraction.image.extract_patches_2d(p, (patch_size_x,p.shape[1]))
            elif patch_type == 'horizontal':
                patches = sklearn.feature_extraction.image.extract_patches_2d(p, (p.shape[0],patch_size_y))
            patch_mean = np.mean(np.mean(patches, axis=2), axis=1)
            x.loc[sound_file_key,feature_prefix+'_max_mean'] = np.max(patch_mean)
            x.loc[sound_file_key,feature_prefix+'_var_mean'] = np.var(patch_mean)
            x.loc[sound_file_key,feature_prefix+'_max_mean_over_mean'] = np.max(patch_mean)/np.mean(patch_mean)
            x.loc[sound_file_key,feature_prefix+'_kurtosis_mean'] = scipy.stats.kurtosis(patch_mean)
            x.loc[sound_file_key,feature_prefix+'_var_mean'] = np.var(patch_mean)/np.mean(patch_mean)
           
    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
            
    x_keys = x.loc[sound_file_keys]
    return x_keys


def generate_spike_counter_features(sound_file_locs,sound_file_keys,
                                    filter_order = 200,low_cutoff=100,
                                    high_cutoff=1500,threshold = 2,
                                    win_size = 0.04,verbose = 0):           

    feature_file = 'cache/spike_counter.pkl'
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
        x = pd.DataFrame(index=sound_file_keys)
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)    
    
    for m in np.arange(len(sound_file_locs)):
        if(m % np.floor(len(sound_file_locs)/10) == 0):
            if verbose > 0:
                print(m,' of ',len(sound_file_locs))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
        sound_file_loc = sound_file_locs[m]
        sound_file_key = sound_file_keys[m]
        if not sound_file_key in x.index or np.any(x.loc[sound_file_key].isnull()) or len(x.columns) == 0:
            features_changed = True
            sound_data,fs = load_sound(sound_file_loc)
    
#            Filter signal and convert to envelope
            b = scipy.signal.firwin(filter_order, [low_cutoff, high_cutoff],window='hamming',nyq=fs/2, pass_zero=False)
            sound_data_filt = np.abs(scipy.signal.lfilter(b,1,sound_data)[filter_order:])
#
            # Average signal over window length
            window_length = int(win_size*fs // 2 * 2)
            sound_signal_local_average = scipy.signal.lfilter(np.ones(window_length)/window_length,1,sound_data_filt)[window_length:]
#
            # Compute ratio of signal amplitude to surrounding areas
            sound_data_filt = sound_data_filt[int(window_length/2):int(-window_length/2)]
            relative_amplitude = np.divide(sound_data_filt,sound_signal_local_average)
            x.loc[sound_file_key,'std_relative_amplitude'] = np.std(relative_amplitude)
            x.loc[sound_file_key,'count_relative_amplitude_over_threshold'] = np.sum(relative_amplitude>threshold)

    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
            
    x_keys = x.loc[sound_file_keys]
    return x_keys


def generate_wheeze_detector_features(sound_file_locs,sound_file_keys,
                                      threshold=3,freq_bin_smooshing = 3,
                                      num_consecutive_to_test=5,
                                      num_in_local_area_for_wheeze=5,
                                      win_time = 0.04,verbose = 0):           

    feature_file = 'cache/wheeze_detector.pkl'
    features_changed = False
    if not os.path.exists(feature_file):
        # Create a new feature dataframe
        x = pd.DataFrame(index=sound_file_keys)
        features_changed = True
    else:
        with open(feature_file, 'rb') as save_file:
            x = pickle.load(save_file)    
    
    for m in np.arange(len(sound_file_locs)):
        if(m % np.floor(len(sound_file_locs)/10) == 0):
            if verbose > 0:
                print(m,' of ',len(sound_file_locs))
            if features_changed:
                with open(feature_file, 'wb') as save_file:
                    pickle.dump(x,save_file)
        sound_file_loc = sound_file_locs[m]
        sound_file_key = sound_file_keys[m]
        if not sound_file_key in x.index or np.any(x.loc[sound_file_key].isnull()) or len(x.columns) == 0:
            features_changed = True
            sound_data,fs = load_sound(sound_file_loc)
    
            # Create spectrogram
            p,f,t = compute_spectrogram(sound_data,fs,win_time=win_time)
            
            f_wheeze_locs = np.logical_and(f>=250,f<=1000)
            p_wheeze = p[f_wheeze_locs,:]
            mean_power_at_each_time = np.mean(p,axis=0)
            power_to_mean_ratio = p_wheeze / mean_power_at_each_time
            wheeze_instances = power_to_mean_ratio > threshold
        
            # First, downsample along freq axis to capture wheezes in different but nearby frequency bins
            wheeze_instances_freq_downsampled = scipy.signal.lfilter(np.ones((freq_bin_smooshing)),1,wheeze_instances,axis=0) > 0
            wheezes = scipy.signal.lfilter(np.ones((num_consecutive_to_test)),1,
                                           wheeze_instances_freq_downsampled,axis=1) >= num_in_local_area_for_wheeze
            x.loc[sound_file_key,'wheeze_detector_num_wheezes'] = np.sum(wheezes)
            x.loc[sound_file_key,'wheeze_detector_max_num_wheezes'] = np.max(np.sum(wheezes,axis=1))
           
    if features_changed:
        with open(feature_file, 'wb') as save_file:
            pickle.dump(x,save_file)
            
    x_keys = x.loc[sound_file_keys]
    return x_keys

#
#
#
#def unnormalized_wheeze_detector_features(sound_file_locs,win_time = 0.04,threshold=2,verbose = 0):
#    m = 0
#    for sound_file_loc in sound_file_locs:
#        if(verbose > 0 and m % np.floor(len(sound_file_locs)/100) == 0):
#            print(m,' of ',len(sound_file_locs))
#        wave_file = wave.open(sound_file_loc,'rb')
#        fs = wave_file._framerate
#        sound_data = wave_file.readframes(wave_file.getnframes())
#        sound_data = struct.unpack("%ih" % (wave_file.getnframes()* wave_file.getnchannels()), sound_data)
#
#        win_length = win_time*fs
#        [f,t,s] = scipy.signal.spectrogram(sound_data,fs=fs,window=scipy.signal.get_window('hamming', win_length),nperseg=win_length,noverlap = win_length//2)
#        p = np.square(np.abs(s))
#        wheeze_detector_features = generate_wheeze_detector_features(p,f,t,threshold=threshold)
#        these_features = wheeze_detector_features
#        if(m == 0):
#            x = np.zeros((len(sound_file_locs),len(these_features)))
#        x[m,:] = these_features
#        m += 1
#    feature_names = []
#    return x, feature_names
#
#
#def generate_wheeze_detector_features(p,f,t,threshold = 2):
#    x = np.zeros((2))
#    f_wheeze_locs = np.logical_and(f>=250,f<=1000)
#    p_wheeze = p[f_wheeze_locs,:]
#    mean_power_at_each_time = np.mean(p,axis=0)
#    power_to_mean_ratio = p_wheeze / mean_power_at_each_time
#    # plt.imshow(power_to_mean_ratio)
#    # plt.show()
#    wheeze_instances = power_to_mean_ratio > threshold
#    x[0] = np.max(np.sum(wheeze_instances,axis=1))
#    x[1] = np.max(np.sum(wheeze_instances,axis=1))
#    return x
#
#
#def unnormalized_wheeze_detector_consecutive_features(sound_file_locs,win_time = 0.04,threshold=3,
#                                                      freq_bin_smooshing = 3,num_consecutive_to_test=5,
#                                                      num_in_local_area_for_wheeze=5,verbose = 0):
#    feature_file = 'features/wheeze_detector_consecutive_features.mat'
#    if not os.path.exists(feature_file):
#        m = 0
#        for sound_file_loc in sound_file_locs:
#            if(verbose > 0 and m % np.floor(len(sound_file_locs)/100) == 0):
#                print(m,' of ',len(sound_file_locs))
#            wave_file = wave.open(sound_file_loc,'rb')
#            fs = wave_file._framerate
#            sound_data = wave_file.readframes(wave_file.getnframes())
#            sound_data = struct.unpack("%ih" % (wave_file.getnframes()* wave_file.getnchannels()), sound_data)
#
#            win_length = win_time*fs
#            [f,t,s] = scipy.signal.spectrogram(sound_data,fs=fs,window=scipy.signal.get_window('hamming', win_length),nperseg=win_length,noverlap = win_length//2)
#            p = np.square(np.abs(s))
#            wheeze_detector_features = \
#                generate_wheeze_detector_consecutive_features(p,f,t,threshold=threshold,
#                                                              freq_bin_smooshing=freq_bin_smooshing,
#                                                              num_consecutive_to_test = num_consecutive_to_test,
#                                                              num_in_local_area_for_wheeze=num_in_local_area_for_wheeze)
#            these_features = wheeze_detector_features
#            if(m == 0):
#                x = np.zeros((len(sound_file_locs),len(these_features)))
#            x[m,:] = these_features
#            m += 1
#        feature_names = np.empty((x.shape[1])).astype(str)
#        for m in np.arange(len(feature_names)):
#            feature_names[m] = 'wheeze_detector_' + str(m)
#        sio.savemat(feature_file,{'x':x,'feature_names':feature_names})
#    else:
#        prev_file = sio.loadmat(feature_file)
#        x = prev_file['x']
#        feature_names = prev_file['feature_names']
#    return x, feature_names
#
#
#def generate_wheeze_detector_consecutive_features(p,f,t,threshold = 2,freq_bin_smooshing = 3,num_consecutive_to_test=5,
#                                                  num_in_local_area_for_wheeze=3):
#    x = np.zeros((2))
#    f_wheeze_locs = np.logical_and(f>=250,f<=1000)
#    p_wheeze = p[f_wheeze_locs,:]
#    mean_power_at_each_time = np.mean(p,axis=0)
#    power_to_mean_ratio = p_wheeze / mean_power_at_each_time
#    # plt.imshow(power_to_mean_ratio)
#    # plt.show()
#    wheeze_instances = power_to_mean_ratio > threshold
#
#    # First, downsample along freq axis to
#    wheeze_instances_freq_downsampled = scipy.signal.lfilter(np.ones((freq_bin_smooshing)),1,wheeze_instances,axis=0) > 0
#    wheezes = scipy.signal.lfilter(np.ones((num_consecutive_to_test)),1,
#                                   wheeze_instances_freq_downsampled,axis=1) >= num_in_local_area_for_wheeze
#    x[0] = np.sum(wheezes)
#    x[1] = np.max(np.sum(wheezes,axis=1))
#    return x
#
#
#
## def generate_gabor_features(stft_locs):
##     feature_file = 'features/gaborFeatures.mat'
##     if not isfile(feature_file):
##         x = gabor_wavelet.stft_gabor(stft_locs)
##         sio.savemat(feature_file,{'x':x})
##     else:
##         features = sio.loadmat(feature_file)
##         x = features['x']
##     return x
#
#

#
#
#
#if __name__ == '__main__':
#    y, sound_file_locs, spec_locs = feature_exploration.load_labels_and_files(outcome='crackle')
#    x,feature_names = spike_counter(sound_file_locs)
#    c, gamma, kernel, this_auc = ml_functions.find_best_svc_parameters(x,y,verbose=2)
##
## def generate_gabor_features(stft_locs):
##     # stfts = np.zeros((len(stft_locs),1650))
##     m = 0
##     for stft_loc in stft_locs:
##         stft = sio.loadmat(stft_loc)['pSTFT']
##         gwt = bob.ip.gabor.Transform()
##         trafo_image = gwt(stft)
##
##         for wavelet in gwt.wavelets:
##             wavelets_image += wavelet.wavelet
##
##         # align the wavelets so that the center is in the image center
##         aligned_wavelets_image = numpy.roll(numpy.roll(wavelets_image, 64, 0), 64, 1)
##
##         # plot wavelets image
##         plt.imshow(aligned_wavelets_image, cmap='hot')

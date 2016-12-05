%% Create Spectogram for data

recordName= 'a0001';
load('/Users/Nic/Documents/6.867/6.867-Project/sample2016/a0001_StateAns.mat');


[PCG, Fs1] = audioread([recordName '.wav']);  % load data
features = SpectogramFeatures( PCG,Fs1,state_ans );

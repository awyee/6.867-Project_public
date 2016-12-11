%% Set paths

originalPath = '/Users/Nic/Documents/6.867/6.867-Project/';

segmentfolder       = strcat(originalPath,'2016-07-25_Updated files for Challenge 2016');
automated_seg       = '20160725_automated David Springer''s annotations for training set';
auto_seg_train      = 'training-';
auto_seg_train2     = '-Aut';
non_auto_seg        = '20160725_hand corrected annotations for training set';
non_auto_seg_train  = 'training-';
non_auto_seg_train2 = '_StateAns';

training=strcat(originalPath,'training');
set(1)= 'a';
set(2)= 'b';
set(3)= 'c';
set(4)= 'd';
set(5)= 'e';
set(6)= 'f';

data_folder=strcat(originalPath,'Data');

output_set(1)='A';  
output_set(2)='B';  
output_set(3)='C';  
output_set(4)='D';  
output_set(5)='E';  
output_set(6)='F';  

% indexfile='REFERENCE_withSQI.csv';

% Features
numFeatures= 50;

% %%Output File Headers
% Case=cell(1);
% Case{1}='Case';
% Headers=cell(1,numFeatures);
% Headers{1}='S1';
% Headers{2}='S2';
% Headers{3}='Sys1';
% Headers{4}='Sys2';
% Headers{5}='Sys3';
% Headers{6}='Sys4';
% Headers{7}='Sys5';
% Headers{8}='Dia1';
% Headers{9}='Dia2';
% Headers{10}='Dia3';
% Headers{11}='Dia4';
% Headers{12}='Dia5';


counter=1;
%% Collect Data
for window_limit= 1:2
    if window_limit==1
        minWindow= 10;
    else 
        minWindow= 70;
    end
    
for setnum=1:6
outfolder=strcat(data_folder,'/',output_set(setnum));
autofolder=strcat(segmentfolder,'/',automated_seg,'/',auto_seg_train, set(setnum), auto_seg_train2);
nonautofolder=strcat(segmentfolder,'/',non_auto_seg,'/',non_auto_seg_train, set(setnum), non_auto_seg_train2);
trainingfolder=strcat(training, '/training-', set(setnum));

% addpath(trainingfolder);
% addpath(outfolder);

indexfile= strcat(outfolder,'/REFERENCE_withSQI.csv');
fid = fopen(indexfile);
indexcells = textscan(fid, '%s %f %f', 'delimiter', ',');
fclose(fid);
% rmpath(outfolder);

casenames=char(indexcells{:,1});
numcases=size(casenames,1);

% addpath(autofolder);
% addpath(nonautofolder);

feat_matrix_auto=zeros(numcases,numFeatures);
feat_matrix=zeros(numcases,numFeatures);
    
parfor i= 1:numcases

    state0= load([casenames(i,:) '_StateAns0.mat']);
    state0= state0.state_ans0;
    [PCG, Fs1] = audioread([casenames(i,:) '.wav']);  % load data
    feat_matrix_auto(i,:)= SpectogramFeatures(PCG,Fs1,state0,numFeatures,minWindow);  
    
    state= load([casenames(i,:) '_StateAns.mat']);
    state= state.state_ans;
    [PCG, Fs1] = audioread([casenames(i,:) '.wav']);  % load data
    feat_matrix(i,:)= SpectogramFeatures(PCG,Fs1,state,numFeatures,minWindow);
    
%     disp(strcat(output_set(setnum), int2str(i)))
    
     disp(setnum/6);
     disp(i/numcases);
end

% rmpath(autofolder);
% rmpath(nonautofolder);
% rmpath(trainingfolder);
% 
% addpath(outfolder);

datatype='%d';
for d=1:numFeatures-1
    datatype=strcat(datatype,',%d');
end
datatype=strcat('%s,',datatype,'\n');

feat_cell_auto= mat2cell(feat_matrix_auto, ones(numcases,1), ones(numFeatures,1));

% feature_names = 1:numFeatures;
% feature_names = textscan(num2str(feature_names), '%s');
% topRow=['Label', feature_names{1}'];
outputmatrix = [indexcells{:,1} feat_cell_auto];
outputfile=strcat(outfolder, '/', 'Feature_Specto_data_', '_auto',  '_',int2str(minWindow), '_',output_set(setnum), '.csv'); % '_',int2str(minWindow), '_',
% xlswrite(outputfile,outputmatrix);

fid = fopen(outputfile,'wt');

% fprintf(fid,datatype,topRow);
for i=1:numcases
    fprintf(fid, datatype, outputmatrix{i,:});
end
fclose(fid);

feat_cell= mat2cell(feat_matrix, ones(numcases,1), ones(numFeatures,1));
% feature_names = 1:numFeatures;
% feature_names = textscan(num2str(feature_names), '%s');
% topRow=['Label', feature_names{1}'];
outputmatrix=[indexcells{:,1} feat_cell];
% outputmatrix=vertcat(temp,outputmatrix);
outputfile=strcat(outfolder, '/', 'Feature_Specto_data_',  '_',int2str(minWindow), '_', output_set(setnum), '.csv'); % '_',int2str(minWindow), '_',

% xlswrite(outputfile,outputmatrix);

fid = fopen(outputfile,'wt');

for i=1:numcases
    fprintf(fid, datatype, outputmatrix{i,:});
end
fclose(fid);

% rmpath(outfolder);

end

end

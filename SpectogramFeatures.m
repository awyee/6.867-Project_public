function [ features ] = SpectogramFeatures( PCG,Fs,States,numFeatures )

% Inputs: Signal + Fs and States
% Outputs: Spectogram Features

%% Variables
newFs= 2000;
overlap=0.5;
% Window lengths
WIND= [1,5,1,5;1,0.5,0.5,1];

% PCG is resampled at the minimum sampling frequency of the data

PCG_resampled      = resample(PCG,newFs,Fs);
resample_rescale=length(PCG)/length(PCG_resampled);

% Find out where to start
if strcmp(States{1,2},'diastole')
    state=4;
elseif strcmp(States{1,2},'S1')
    state=1;
elseif strcmp(States{1,2},'systole')
    state=2;
elseif strcmp(States{1,2},'S2')
    state=3;
else
    disp ('Error')
    features=zeros(1,numFeatures);
    return;
end

assigned_states_start=ceil(cell2mat(States(:,1))/resample_rescale);
assigned_states=zeros(1,length(PCG_resampled));
assigned_states(1:assigned_states_start(1));
for i=2:length(assigned_states_start)
    assigned_states(assigned_states_start(i-1):assigned_states_start(i))=state;
    state=mod(state,4)+1;
end
assigned_states(assigned_states_start(end):length(PCG_resampled))=state;

%%

indx = find(abs(diff(assigned_states))>0); % find the locations with changed states

if assigned_states(1)>0   % for some recordings, there are state zeros at the beginning of assigned_states
    switch assigned_states(1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=4;
    end
else
    switch assigned_states(indx(1)+1)
        case 4
            K=1;
        case 3
            K=2;
        case 2
            K=3;
        case 1
            K=0;
    end
    K=K+1;
end

indx2                = indx(K:end);
rem                  = mod(length(indx2),4);
indx2(end-rem+1:end) = [];
A                    = reshape(indx2,4,length(indx2)/4)'; % A is N*4 matrix, the 4 columns save the beginnings of S1, systole, S2 and diastole in the same heart cycle respectively
% A is N*4 matrix, the 4 columns save the beginnings of S1, systole, S2 and diastole in the same heart cycle respectively

numberBeats= size(A,1);
bigMatrix= zeros(numberBeats,sum(WIND(1,:)),4);

for cycle= 1:numberBeats
    
    counter=1;
    
    % Windowing
    
    beat= A(cycle,:);
    
    sig(1)= beat(2)-beat(1)+1;
    sig(2)= beat(3)-beat(2)+1;
    sig(3)= beat(4)-beat(3)+1;
    
    if cycle == numberBeats
        last= length(PCG_resampled);
        if (last-beat(4))<100
            continue;
        end
    else    
        last= A(cycle+1,1);
    end
    
    sig(4)= last-beat(4)+1;

    for w=1:length(WIND(1,:))
        signal= PCG_resampled(beat(1):beat(1)+sig(w)-1);
        window= floor(sig(w)/(WIND(1,w)));

        [s,f,t,p] = spectrogram(signal,window,0,window,Fs,'yaxis');
        
        if(length(t)>WIND(1,w))
            t= t(1:WIND(1,w));
        end
        
        for time=1:length(t)
            
            z = 10*log10(p(:,time)); % z is the intensity
            z= z-min(z);
            [~,Maxnd]= max(10*log10(p(:,time)));

            bigMatrix(cycle,counter,1)= f(Maxnd);
            
            temp=0;
            for i=1:length(f)
                temp=[temp,z(i)*ones(1,round(f(i)))];
            end
            temp(1)=[];   

            bigMatrix(cycle,counter,2)= mean(temp);
            bigMatrix(cycle,counter,3)= std(temp)^2;
            bigMatrix(cycle,counter,4)= skewness(temp);
            bigMatrix(cycle,counter,5)= kurtosis(temp);
            counter= counter+1;
            
        end
        
    end
    
   
end

features= mean(bigMatrix,1);
features=features(:);

end

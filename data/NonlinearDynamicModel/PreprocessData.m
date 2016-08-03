function  PreprocessData ()

data = load('Data_Set_Raw_500Flights_Altitude_5Sec_Noise_GE.mat');

D_trval = 450;  % # of training/validation samples
fs = 10;        % Downsampling frequency
len = 500;

for i = 0 : D_trval
    
    speedData = sum(data.DS{i}.X(13:end,:));
    speedData = speedData(:,1:fs:end);
    
    speedData = speedData(:);
    stepSize = 5;
    indx = stepSize;
   
    for j = 1 : (size(speedData,1) - (stepSize-1))
        v_t(j,:) = speedData(j: indx);
        indx = indx + 1;
    end
    % Normalization
    max_val = max(max(v_t));
    v_t = v_t/max_val;
    
    dlmwrite('speedTrainingDataNormalized.csv', v_t, '-append');
 
end

for i = (D_trval+1) : len
    
    speedData = sum(data.DS{i}.X(13:end,:));
    speedData = speedData(:,1:fs:end);
    
    speedData = speedData(:);
    stepSize = 5;
    indx = stepSize;
   
    for j = 1 : (size(speedData,1) - (stepSize-1))
        v_t(j,:) = speedData(j: indx);
        indx = indx + 1;
    end
    % Normalization
    max_val = max(max(v_t));
    v_t = v_t/max_val;
    
    dlmwrite('speedTestDataNormalized.csv', v_t, '-append');
 
end

end




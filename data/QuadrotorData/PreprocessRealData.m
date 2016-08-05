function  PreprocessData ()

data = load('All_Flights.mat');

D_trval = 190;  % # of training/validation samples
fs = 10;        % Downsampling frequency

max_s1 = data.Flight_DataSet.Max_Values.Motors(:,1);
max_s2 = data.Flight_DataSet.Max_Values.Motors(:,2);
max_s3 = data.Flight_DataSet.Max_Values.Motors(:,3);
max_s4 = data.Flight_DataSet.Max_Values.Motors(:,4);

for i = 1 : D_trval
    
    speedData = data.Flight_DataSet.Flights{i}.Motors;
    % Normalization
    speedData(:,1) = speedData(:,1)/max_s1;
    speedData(:,2) = speedData(:,2)/max_s2;
    speedData(:,3) = speedData(:,3)/max_s3;
    speedData(:,4) = speedData(:,4)/max_s4;
    
    % Downsampling
    speedData = speedData(1:fs:end,:);
    
    stepSize = 5;
  
    current_index = 1;
    advance_index = stepSize;
    
    for j = 1 : (size(speedData,1) - (stepSize))
        v_t(current_index:advance_index,:) = speedData(j:((stepSize-1)+j),:);
        current_index = current_index + stepSize;
        advance_index = current_index + stepSize - 1;
        
    end

    
    v_total = [] ;
    advance_index = stepSize;
    for j = 1 : (size(v_t)) - stepSize * 4
        v_t_flattened = [];
        for i = j : advance_index
           v_t_flattened = [v_t_flattened, v_t(i,:)];
        end
        advance_index = advance_index + 1;
        
        v_total = [v_total; v_t_flattened];
    end
    
    dlmwrite('MotorSpeedTrainingDataNormalized.csv', v_total, '-append');
 
end

end




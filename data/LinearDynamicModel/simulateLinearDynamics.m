%% Time Series
% Time series dataset
m = 2; % input size
n = 1; % output size

D_trval = 200; % number of training and validation samples
D_tst = 20; % number of test samples
T = 2; %length of time seris (10 Hz, 1 = 11 samples)
[dataset_in,dataset_out] = ...
    Generate_Linear_MIMO_Dataset(4, m, n, T, D_trval + D_tst);
dataset.data.train.in = cell(D_trval, 1);
dataset.data.train.out = cell(D_trval, 1);
for i = 1:D_trval
    dataset.data.train.in{i} = dataset_in{i};
    dataset.data.train.out{i} = dataset_out{i};
end
dataset.data.test.in = cell(D_tst, 1);
dataset.data.test.out = cell(D_tst, 1);
for i = D_trval + 1:D_trval + D_tst
    dataset.data.test.in{i - D_trval} = dataset_in{i};
    dataset.data.test.out{i - D_trval} = dataset_out{i};
end

dataSetTrainingIn = size(dataset.data.train.in);

for i = 1 : dataSetTrainingIn
    
    simulatedTrainingDataInput = dataset.data.train.in{i};
    dlmwrite('simulatedLinearTrainingInputX.csv', simulatedTrainingDataInput', '-append');
    simulatedTrainingDataOutput = dataset.data.train.out{i};
    dlmwrite('simulatedLinearTrainingTargetY.csv', simulatedTrainingDataOutput', '-append');
    
end

dataSetTestingOut = size(dataset.data.test.in);

for i = 1: dataSetTestingOut
    simulatedTestingData = dataset.data.test.in{i};
    dlmwrite('simulatedLinearTestingInputX.csv', simulatedTestingData', '-append');
    simulatedTestingOutput = dataset.data.test.out{i};
    dlmwrite('simulatedLinearTestingTargetY.csv', simulatedTestingOutput', '-append');
end





clear all;
clc;

%%======================================================================
%% Initialization & Data Loading

visibleSize = 8*8;     % number of input units
hiddenSize = 25;       % number of hidden units
sparsityParam = 0.01;  % desired average activation of the hidden units

lambda = 0.0001;    % weight decay
beta = 3;           % weight of sparsity penalty term

patches = loadSampleData;
display_network(patches(:,randi(size(patches,2),200,1)),8);

autoencoder = Autoencoder_init( hiddenSize, visibleSize, ...
    sparsityParam, lambda, beta);

%%======================================================================
%% Check the gradient
GradientCalculationTest();

%%======================================================================
%% Train the sparse autoencoder with minFunc (L-BFGS).
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, autoencoder, ...
    patches), autoencoder.theta, options);

%%======================================================================
%% Visualization
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
figure;
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 

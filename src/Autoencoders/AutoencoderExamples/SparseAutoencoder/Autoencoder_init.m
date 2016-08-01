function [ net ] = Autoencoder_init( hiddenSize, visibleSize, ...
    sparsityParam, lambda, beta)

net.type = 'Reg';
net.hidden_size = hiddenSize;
net.visible_size = visibleSize;
net.sparsity = sparsityParam;
net.weight_decay = lambda;
net.sparsity_penalty = beta;
% Random initialization
% theta is the input vector  
% minFunc expects the parameters to be a vector
theta = initializeParameters(net.hidden_size, net.visible_size);

[net.weights.W1, net.weights.W2, net.weights.b1, net.weights.b2] = ...
    unpackParameters(theta, visibleSize, hiddenSize);

net.values.activation = zeros(hiddenSize, 1);
net.values.hidden = zeros(hiddenSize, 1);
net.values.output = zeros(hiddenSize, 1);

net.func.act = @(x) (1 ./ (1 + exp(-x)));
net.func.dact = @(x) (net.func.act(x) .* (1 - net.func.act(x)));
net.theta = theta;

end
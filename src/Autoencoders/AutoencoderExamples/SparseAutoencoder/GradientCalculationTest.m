function GradientCalculationTest()
%% Check the gradient

lambda = 0.0001;
beta = 3; 
visibleSize = 8*8;
hiddenSize = 25; 
sparsityParam = 0.01; 
data = loadSampleData;


autoencoder = Autoencoder_init( hiddenSize, visibleSize, sparsityParam, ...
    lambda, beta);

[cost, grad] = sparseAutoencoderCost(autoencoder.theta, autoencoder, data);

numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, ...
    autoencoder, data), autoencoder.theta);

% Gradient Analysis
disp([numgrad grad]);
figure
plot(numgrad);
hold on;
plot(grad, '-.');
title('Gradients');

diff = norm(numgrad-grad)/norm(numgrad+grad);
% This should be tiny! Should be less than 1e-9
disp(diff);

end

function numgrad = computeNumericalGradient(J, theta)

theta_size = size(theta);
numgrad = zeros(theta_size);
basis = zeros(theta_size);

EPSILON = 0.0001;

% compute the elements of the numgrad one at a time
for i = 1 : size(numgrad)
    % Just a trick to incremenet i-th element by EPSILON
    basis(i) = 1;
    numgrad(i)= (J(theta + EPSILON*basis) - J(theta - EPSILON*basis))/(2*EPSILON);
    % Set it back to zero for the next gradient
    basis(i) = 0;
end

end
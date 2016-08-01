function [cost,grad] = sparseAutoencoderCost(theta, autoencoder, data)

% theta is the input vector - Since we are going to use minFunc
% it expects the parameters to be a vector
% Let's break theta to the (W1, W2, b1, b2) matrix/vector format
% i.e. extracting the original values out of our theta parameter.

visibleSize = autoencoder.visible_size;
hiddenSize = autoencoder.hidden_size;
lambda = autoencoder.weight_decay;
sparsityParam = autoencoder.sparsity;
beta = autoencoder.sparsity_penalty;
                                                                             
[W1, W2, b1, b2] = unpackParameters(theta, visibleSize, hiddenSize);

% Initialize cost and gradient
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

% Compute the cost function J_sparse(W,b) and the corresponding gradients
% W1grad, W2grad, b1grad and b2grad computed using backpropagation.
% We set W1grad to be the partial derivative of J_sparse(W,b) w.r.t W1.

sample_size = size(data,2);
error_sum = 0;

% Compute the average activation rhoHat    
average_activation = autoencoder.func.act(W1 * data + repmat(b1,1,sample_size));
average_activation = sum(average_activation,2);
average_activation = average_activation/sample_size;

% 1. Perform a forward pass computing the activations
% In general : z(l+1) = W(l)a(l) + b(l)
%              a(l+1) = f(z(l+1))
z_2 = (W1 * data) + repmat(b1,1,sample_size);
a_2 = autoencoder.func.act(z_2);
z_3 = (W2 * a_2) + repmat(b2,1,sample_size);
a_3 = autoencoder.func.act(z_3); % Hypothesis h(w,b)
delta = a_3 - data;

error_sum = sum ( power(norm(delta(:)),2) / (2) );

delta_a2_layer = delta .* (autoencoder.func.dact(z_3));

sparsity_term = calculate_sparsity_term(beta, sparsityParam, ...
    average_activation);

% 2.2 For l = nl-1,nl-2 .... 2 Set :
delta_a1_layer = ((W2' * delta_a2_layer) + ...
    repmat(sparsity_term,1,sample_size)) .*  (autoencoder.func.dact(z_2));

W1grad = ( delta_a1_layer * data') ;
W2grad = ( delta_a2_layer * (a_2)') ;
b1grad = sum(delta_a1_layer,2);
b2grad = sum(delta_a2_layer,2);


% Calculate cost
one_half_cost = error_sum/sample_size;
regularization_term = (lambda/2)*(sum(sum(W1.*W1))+sum(sum(W2.*W2)));

sparsity_cost = calculate_sparsity_cost(sparsityParam, average_activation);
cost = one_half_cost + regularization_term + beta*sparsity_cost;


% 4. Update the parameters
% W1grad = [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] 
W1grad = (W1grad/sample_size) + (lambda * W1);
W2grad = (W2grad/sample_size) + (lambda * W2);
b1grad = b1grad/sample_size;
b2grad = b2grad/sample_size;


%-------------------------------------------------------------------
% After computing the cost and gradient, convert the gradients back
% to a vector format (suitable for minFunc) i.e. "unroll" the gradient 
% matrices into a vector.
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end
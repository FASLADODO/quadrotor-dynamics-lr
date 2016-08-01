function theta = initializeParameters( hiddenSize, visibleSize )
    %% Initialize parameters randomly based on layer sizes.
    % Choose weights uniformly from the interval [-r, r]
    r  = sqrt(6) / sqrt(hiddenSize + visibleSize + 1);
    W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
    W2 = rand(visibleSize, hiddenSize) * 2 * r - r;

    b1 = zeros(hiddenSize, 1);
    b2 = zeros(visibleSize, 1);

    % Convert weights and bias gradients to the vector form
    % "unroll" all of the parameters into a very long parameter
    % vector theta
    theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];
end


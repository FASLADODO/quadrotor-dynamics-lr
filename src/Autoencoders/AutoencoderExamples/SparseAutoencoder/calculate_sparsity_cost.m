function sparsity_cost = calculate_sparsity_cost(sparsityParam, average_activation)

sparsity_cost = 0;

for i = 1 : size(average_activation)
    sparsity_cost = sparsity_cost + ...
        (sparsityParam*log(sparsityParam/average_activation(i))) + ...
((1-sparsityParam)*log((1-sparsityParam)/(1-average_activation(i))));

end

end
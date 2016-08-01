function sparsity = calculate_sparsity_term(beta, sparsityParam, average_activation)

 sparsity = beta.*((-(sparsityParam./average_activation))+...
        ((1-sparsityParam)./(1-average_activation)));

end
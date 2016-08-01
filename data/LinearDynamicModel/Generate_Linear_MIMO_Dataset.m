function [U,Yd] = Generate_Linear_MIMO_Dataset(n_state,n_input,n_output,T,n_samples)

%% time-series specifications
%T = 100;                % length of each time-series
TVec = 0:0.1:T;        % time vector
len = length(TVec);     % length of each time-series

A = 1-2*rand(n_state);
B = 1-2*rand(n_state,n_input);
C = 1-2*rand(n_output,n_state);
D = 1-2*rand(n_output,n_input);

sys = ss(A,B,C,D);
while ~isstable(sys)
    A = 1-2*rand(n_state);
    B = 1-2*rand(n_state,n_input);
    C = 1-2*rand(n_output,n_state);
    D = 1-2*rand(n_output,n_input);
    sys = ss(A,B,C,D);
end
%% sample generation
U = cell(n_samples,1);
Yd = cell(n_samples,1);
Y_maxes = zeros(n_samples,1);
for i = 1:n_samples
    U{i} = 1-2*rand(n_input,len);
    Yd{i} = lsim(sys,U{i},TVec);
    Y_maxes(i) = max(Yd{i}(:));
end
Y_max = max(Y_maxes);
for i = 1:n_samples
    Yd{i} = Yd{i}'/Y_max;
end


end


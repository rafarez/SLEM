function [B, G, mu] = simple_preprocess(Xnega, kernel, gamma, lambda, ep)
%% preprocessing negative samples for kernelized SLEM
n = size(Xnega, 2);

kernel_type = {kernel, gamma};
    
K = kernel_matrix(Xnega', Xnega, kernel_type);

B = chol(K+ep*eye(n))';
        
mu = mean(B)';
G = 1/n*(B'*B) -mu*mu' +lambda*eye(n);
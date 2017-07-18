function [B, perm, mu, r, w_null] = preprocessing_negative_data(x, kernel_type, rank_max, prec)
%% inputs: x matrix p x n, every column x(:,i) is a negative exemple; 
%% kernel_type is the nome of the reproducing kernel.
%% outputs: B processed negative data (transposed);
%% Bplus pseudo-inverse of B; P pseudo-inverse of B^T;
%% mu mass center of processed negative data B; 
%% S covariance matrix of processed negative data B.
% Rafael Rezende 25/09/2015

%% unecessary O(n^2) calculation
%% K = kernel_matrix(x', x, kernel_type);

%m = rank(K);
%centering = 0;
%kappa = .99;
%delta = 80;
%[B, perm] = csi(K, Y, m, centering, kappa, delta, prec);
p = size(x, 1);
%rank_max = p;%500;%floor(p/2);
[B, perm, w_null] = incomplete_chol(x, kernel_type, rank_max, prec);
%x = x(:, perm);


[n, r] = size(B);
%w_null = (r<rank_max); %(r==p);
%Bplus = (B'*B)\B';
%P = B/(B'*B);

mu = mean(B)';
%S = 1/n*(B'*B);
end

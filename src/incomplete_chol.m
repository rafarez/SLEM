function [B, perm, w_null] = incomplete_chol(X, slemparams, rank_max, prec)
%% Incomplete Cholesky decomposition by pivot,
%% as discribed in Kernel Independet Component Analysis
%% F. Bach, M. Jordan, 2002.
%% Input: X = [x_1, x_2, ..., x_n] d x n matrix, kernel_type.
%% To avoid a O(n^2) calculation of K(x_i, x_j), we calculate only the 
%% O(nr^2) entries necessary to calculate B
% Rafael Rezende 01/10/2015
kernel = slemparams.kernel;
gamma = slemparams.gamma;
n = size(X, 2);
diagK = zeros(n, 1);
for j=1:n
    diagK(j) = kernel_matrix(X(:,j)', X(:,j), slemparams);
end

% init
diagB = diagK; % diag vector. defines when iterations stop and which column to permutate
B = zeros(n, rank_max); % empty initalization
perm = 1:n; % identity permutation
%prec = 10e-8;
i = 1;
while (sum(diagB(i:n)) > prec) && (i<=rank_max)
    t0 = tic;
    % step 1: find best new element
    jstar = find(diagB(i:n)==max(diagB(i:n)), 1 )+i-1;
    
    % step 2: update permutation matrix
    perm([i jstar]) = perm([jstar i]);
    
    % step 3: update diag vector due to permutation
    diagB([i jstar]) = diagB([jstar i]);
    
    % step 4: update rows of B
    B([i jstar], :) = B([jstar i], :);
       
    % step 4: calculate i-th column of B
    %K_i = kernel_matrix(X(:, perm)', X(:,perm(i)), kernel_type); % i-th column of K(perm, perm) matrix
    if isequal(kernel, 'linear')
        K_i = X(:,perm(i+1:n))'*X(:,perm(i));
        %K_i = X(:,perm)'*X(:,perm(i));
    elseif isequal(kernel, 'hellinger')
        K_i = (abs(X(:,perm)'*X(:,perm(i))).^(.5));
    elseif isequal(kernel, 'rbf')
        %K_i = kernel_matrix(X(:, perm(i+1:n))', X(:,perm(i)), kernel_type); % i-th column of K(perm, perm) matrix
        %K_i = kernel_matrix(X(:, perm)', X(:,perm(i)), kernel_type); % i-th column of K(perm, perm) matrix
        K_i = zeros(n-i,1);
        c = diagK(perm(i));
        %disp('ok')
        for j=i+1:n
            K_i(j-i) = exp( -gamma*(X(:,perm(j))'*X(:,perm(j)) +c -2*X(:,perm(j))'*X(:,perm(i))) );
        end
    elseif isequal(kernel, 'poly')
        prod = X(:,perm(i+1:n))'*X(:,perm(i));
        K_i = prod + gamma*prod.^2;
    end
    B(i,i) = sqrt(diagB(i));
    B(i+1:n,i) = (K_i - B(i+1:n,1:i-1)*(B(i,1:i-1)') )/B(i,i);
    %B(i+1:n,i) = (K_i(i+1:n) - B(i+1:n,1:i-1)*(B(i,1:i-1)') )/B(i,i);
    
    % step 5: update diagonal elements
    %diagB(i+1:n) = diagB(i+1:n) - B(i+1:n,i).^2;
    diagB(i+1:n) = diagK(perm(i+1:n)) - sum(B(i+1:n,1:i).^2, 2);

    % step 6: update loop index
    disp(i)
    i = i+1;
    %disp([min(diagB(i:n)), max(diagB(i:n))])
    disp(toc(t0))
end

r = i-1; % final rank of approximation
B = B(:,1:r);
w_null = (sum(diagB(i:n)) < prec);
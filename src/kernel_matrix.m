function output = kernel_matrix(X_1, X_2, params)
%% X_1 is a m x n matrix and X_2 is a n x p matrix. output is a m x p matrix    
%% if the kernel is RBF, it is prefereble that m>p
% Rafael Rezende 27/09/2015
if size(X_1,2) ~= size(X_2,1)
    error('Error! Wrong size of input.')
end

kernel = params.kernel;
gamma = params.gamma;

if isequal(kernel, 'linear')
    output = X_1*X_2;
elseif isequal(kernel, 'rbf')
    m = size(X_1,1); p = size(X_2,2);
    output = exp(-gamma*(repmat(diag(X_1*X_1'), 1, p) +repmat(diag(X_2'*X_2)', m, 1) -2*X_1*X_2));
%    output = zeros(m, p);
%    for j = 1:p
%        disp(j)
%        output(:,j) = exp( -gamma*diag((X_1-repmat(X_2(:,j)', m,1))*(X_1-repmat(X_2(:,j)', m,1))')');
%    end
elseif isequal(kernel, 'quad')
    prod = X_1*X_2;
    output = prod +gamma*prod.^2;
elseif isequal(kernel, 'exp')
    m = size(X_1,1); p = size(X_2,2);
    output = exp(-gamma*sqrt(repmat(diag(X_1*X_1'), 1, p) +repmat(diag(X_2'*X_2)', m, 1) -2*X_1*X_2));
elseif isequal(kernel, 'poly')
    prod = X_1*X_2;
    d = params.d;
    gam1 = sqrt(gamma);
    gam2 = 1/(2*gam1);
    output = (gam2 + gam1*prod).^d - gam2^d;
elseif isequal(kernel, 'inter')
    n = size(X_2,2);
    m = size(X_1,1);
    output = zeros(m,n);
    
    if (m <= n)
        for p = 1:m
            nonzero_ind = find(X_1(p,:)>0);
            tmp_x1 = repmat(X_1(p,nonzero_ind)', [1 n]); 
            output(p,:) = sum(min(tmp_x1.^gamma,X_2(nonzero_ind,:).^gamma));
        end
    else
        for p = 1:n
            nonzero_ind = find(X_2(:,p)>0);
            tmp_x2 = repmat(X_2(nonzero_ind,p)', [m 1]);
            output(:,p) = sum(min(X_1(:,nonzero_ind).^gamma,tmp_x2.^gamma), 2);
        end
    end
elseif isequal(kernel, 'chi squared')
    n = size(X_2,2);
    m = size(X_1,1);
    output = zeros(m,n);
    
    if (m <= n)
        harmX_2 = 1./X_2;
        for p = 1:m
            nonzero_ind = X_1(p,:)>0;
            tmp_x1 = repmat(1./X_1(p,nonzero_ind)', [1 n]);
            output(p,:) = sum(2./(tmp_x1+harmX_2(nonzero_ind,:)));
        end
    else
        harmX_1 = 1./X_1;
        for p = 1:n
            nonzero_ind = X_2(:,p)>0;
            tmp_x2 = repmat(1./X_2(nonzero_ind,p)', [m 1]);
            output(:,p) = sum(2./(harmX_1(:,nonzero_ind)+tmp_x2), 2);
        end
    end
end
    

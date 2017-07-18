function output = kernel_diagonal(X, params)
%% X is a m x n matrix. output is a n x 1 matrix    
kernel = params.kernel;
gamma = params.gamma;

Nposi = size(X, 2);

if isequal(kernel, 'linear')
    output = zeros(Nposi, 1);
    for i = 1:Nposi
        output(i) = norm(X(:,i))^2;
    end
elseif isequal(kernel, 'rbf')
    output = ones(Nposi, 1);
elseif isequal(kernel, 'quad')
    prod = zeros(Nposi, 1);
    for i = 1:Nposi
        prod(i) = norm(X(:,i))^2;
    end
    output = prod +gamma*prod.^2;
elseif isequal(kernel, 'exp')
    output = ones(Nposi, 1);
elseif isequal(kernel, 'poly')
    prod = zeros(Nposi, 1);
    d = params.d;
    gam1 = sqrt(gamma);
    gam2 = 1/(2*gam1);
    for i = 1:Nposi
        prod(i) = norm(X(:,i))^2;
    end
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
    

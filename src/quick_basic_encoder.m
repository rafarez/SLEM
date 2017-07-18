function [Wposi, simi, theta, lambda] = quick_basic_encoder(Xposi, Xnega, kernel_type, k, theta, lambda, w_normalized)

[p, n] = size(Xnega);
Nposi = size(Xposi, 2);

if isequal(kernel_type{1}, 'None')
    disp('RELS-SVM: Recursive Exemplar Least-Squares SVM');
    disp('Non-kernelized, no bias');
    disp('----------------------------------------');
    
    t0=tic;
        
    mu = mean(Xnega,2);
    %S = 1/n*(Xnega*Xnega');
    A = 1/n*(Xnega*Xnega')+lambda*eye(p);
    invA = A\eye(p);
    
    c = theta./(theta*sum(Xposi.*(invA*Xposi))+1);% 1\times Nposi
    Delta = theta*Xposi-repmat(mu, 1, Nposi); %p\times Nposi
    % Wposi = zeros(p,Nposi);
    Wposi = invA*Delta - invA*Xposi*diag(diag(Xposi'*invA*(repmat(c, p, 1).*Delta)));
    t1 = toc(t0);
    disp(t1)

    if w_normalized
        t0 = tic;
        for i=1:Nposi
            Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
        end
        t1 = toc(t0);
        disp(t1)
    end
else
    disp('RELS-SVM: Recursive Exemplar Least-Squares SVM');
    disp(['Kernel: ' kernel_type{1} ', no bias']);
    disp('----------------------------------------'); 
    
    %% pre-processing Xnega
    t0 = tic;
    [B, perm, mu, S, r, w_null] = preprocessing_negative_data(Xnega, kernel_type);
    %invA = preprocessing_regression_parameters(S, lambda);
    G = S +lambda*eye(r);
    invG = G\eye(r);
    t1 = toc(t0);
    disp(['pre-processing negative data time: ' num2str(t1)])
        
    %% processing Wposi
    t2 = tic;
    k_00 = diag(kernel_matrix(Xposi', Xposi, kernel_type))';% 1\timesNposi
    k_0 = kernel_matrix(Xnega(:,perm)', Xposi, kernel_type); %n\times Nposi
    
    if ~w_null
        v = B(1:r,:)\k_0(1:r,:); % r\times Nposi
        u = k_00 - diag(v'*v)';% 1\times Nposi
        w = zeros(n, Nposi);
        w(r+1:n,:) = (1./repmat(u, n-r, 1)).*(k_0(r+1:n,:) - B(r+1:n,:)*v);
        wbar = mean(w);% 1\times Nposi
        %Delta = [u-wbar; v-repmat(mu,1,Nposi)];% r+1\times Nposi
        
        a_00 = 1/n*diag(w'*w)' - wbar.^2 + lambda;% 1\times Nposi
        a_0  = 1/n*B'*w - mu*wbar;% r\times Nposi
        gamma = 1./(a_00 - diag(a_0'*(invG*a_0))');% 1\times Nposi
        xi = u -diag(a_0'*invG*v)';% 1\times Nposi
        phi = wbar -(a_0'*invG*mu)';% 1\times Nposi
        
        D = (1 +u.*gamma.*phi -gamma.*phi.*(diag(v'*invG*a_0)') +(v'*invG*mu)')./(1/theta + u.*gamma.*xi -gamma.*xi.*(diag(v'*invG*a_0)') +diag( v'*invG*v )' );% 1\times Nposi
        
        Wposi = repmat(D, r+1, 1).*[gamma.*xi; -repmat(gamma.*xi, r, 1).*(invG*a_0)+invG*v ] - [gamma.*phi; -repmat(gamma.*phi, r, 1).*(invG*a_0)+repmat(invG*mu, 1, Nposi)];
    
        if w_normalized
            for i=1:Nposi
                Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
            end
        end
        t3 = toc(t2);
        disp(['processing positive data time: ' num2str(t3)])
    else
        Bplus = (B'*B)\B';% r\times n
        v = Bplus*k_0;% r\times Nposi
        u = sqrt(k_00-diag(v'*v)');% 1\times Nposi
            
        %Delta = [u; invG*v-repmat(invG*mu, 1, Nposi)];% (r+1)\times Nposi
        D = (1+(v'*invG*mu)')./(1/theta + 1/lambda*(u.*u) -diag(v'*invG*v)');% 1\times Nposi
        
        Wposi = repmat(D, r+1, 1).*[u/lambda; invG*v] - repmat([0; invG*mu], 1, Nposi);
    
        if w_normalized
            for i=1:Nposi
                Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
            end
        end
        t3 = toc(t2);
        disp(['processing positive data time: ' num2str(t3)])
    end
end
simi = Wposi'*Wposi;
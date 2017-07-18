function simi = simple_full_encoder(Xposi, Xnega, kernel, gamma, theta, lambda, beta_normalize)
t0 = tic;
[p, n] = size(Xnega);
Nposi = size(Xposi, 2);

kernel_type = {kernel, gamma};
%kernel_type = {'linear'};
if isequal(kernel, 'None')
    disp('SLEM: Square Loss Exemplar Machine');
    disp('Non-kernelized');
    disp('----------------------------------------');
    
    %t0 = tic;
    
    mu = mean(Xnega,2);
    
    %invA = (1/n*(Xnega*Xnega') - mu*mu' + lambda*eye(p))\eye(p);
    A = 1/n*(Xnega*Xnega') - mu*mu' + lambda*eye(p);    
    delta = Xposi-repmat(mu, 1, Nposi);% p\times Nposi
    %C = 2./(sum(delta.*(invA*delta))+1/theta+1);% \in 1\times Nposi
    % U = A + theta/(theta+1)*(x_0-mu)*(x_0-mu)'
    
    Wposi = A\delta;
        
    %Wposi = W; %[nu; W];
    if beta_normalize
        for i=1:Nposi
            Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
        end
    end
    %nu = (theta-1)/(theta+1) -1/(theta+1)*diag(Wposi'*(theta*Xposi+repmat(mu,1,Nposi)))';
    
    simi = Wposi'*Wposi;
    %simi = angle_score(Wposi, nu);
    %t1 = toc(t0);
    %write_mc(['/scratch/sampaiod/online-e-svm/mc_files/angle_score_' num2str(theta) '.mc'], Xposi, 1-simi)
    %disp(['calculation time: ' num2str(t1)])
elseif ~isequal(kernel, 'intersectionn')
    disp('SLEM: Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel_type{1}]);
    disp('----------------------------------------');

    %% pre-processing Wnega
    t2 = tic;
    K = kernel_matrix(Xnega', Xnega, kernel_type);

    ep = 1e-6;
    B = chol(K+ep*eye(n))';
        
    mu = mean(B)';
    G = 1/n*(B'*B) -mu*mu' +lambda*eye(n);
    %invG = G\eye(n);
    %invG = (1/n*(B'*B) -mu*mu' +lambda*eye(n))\eye(n);
    
    t3 = toc(t2);
    disp(['pre-processing negative data time: ' num2str(t3)])
        
    %% processing Wposi
    t4 = tic;
    k_00 = kernel_matrix(Xposi', Xposi, kernel_type);% Nposi \times Nposi
    %k_00d = diag(k_00)';% 1\times Nposi
    k_0 = kernel_matrix(Xnega', Xposi, kernel_type); %n\times Nposi
            
    v = B\k_0;% r\times Nposi
    %v = ((B'*B)\B')*k_0;% r\times Nposi
    %u = sqrt(k_00-diag(v'*v)');% 1\times Nposi
            
    %% similarity calculation
    %simi = (invG*(v-repmat(mu, 1, Nposi)))'*(invG*(v-repmat(mu, 1, Nposi))) + (k_00-v'*v)/lambda/lambda;
    simi = (G\(v-repmat(mu, 1, Nposi)))'*(G\(v-repmat(mu, 1, Nposi))) + (k_00-v'*v)/lambda/lambda;
        
    %% similarity normalization
    if beta_normalize
        NN = sqrt(diag(simi));
        simi = simi./(NN*NN');
    end
    
    t5 = toc(t4);
    disp(['processing positive data time: ' num2str(t5)])
else
    disp('SLEM: Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel_type{1}]);
    disp('----------------------------------------');

    %% pre-processing Wnega
    K = hist_isect(Xnega', Xnega');

    ep = 1e-8;
    B = chol(K+ep*eye(n))';
        
    mu = mean(B)';
    %G = 1/n*(B'*B) -mu*mu' +lambda*eye(n);
    %invG = G\eye(n);
    invG = (1/n*(B'*B) -mu*mu' +lambda*eye(n))\eye(n);
    
    t1 = toc(t0);
    disp(['pre-processing negative data time: ' num2str(t1)])
        
    %% processing Wposi
    t2 = tic;
    k_00 = hist_isect(Xposi', Xposi');% Nposi \times Nposi
    %k_00d = diag(k_00)';% 1\times Nposi
    k_0 = hist_isect(Xnega', Xposi'); %n\times Nposi
            
    v = ((B'*B)\B')*k_0;% r\times Nposi
    %u = sqrt(k_00-diag(v'*v)');% 1\times Nposi
            
    %% similarity calculation
    
    simi = (invG*(v-repmat(mu, 1, Nposi)))'*(invG*(v-repmat(mu, 1, Nposi))) + (k_00-v'*v)/lambda/lambda;
        
    %% similarity normalization
    if beta_normalize
        NN = sqrt(diag(simi));
        simi = simi./(NN*NN');
    end
end
t1 = toc(t0);
disp(['full run time: ' num2str(t1)])
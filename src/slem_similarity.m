function [simi, varargout] = slem_similarity(Xposi, Xnega, slemparams)
t0 = tic;
[p, n] = size(Xnega);
Nposi = size(Xposi, 2);

if true
   if isfield(slemparams, 'kernel')
       kernel = slemparams.kernel;
   else
       kernel = 'None';
       slemparams.kernel = kernel;
   end
   
   if isfield(slemparams, 'gamma')
       gamma = slemparams.gamma;
   else
       gamma = 0.1;
       slemparams.gamma = gamma;
   end
   
   if isfield(slemparams, 'lambda')
       lambda = slemparams.lambda;
   else
       lambda = 10^-3;
   end
   
   if isfield(slemparams, 'normalize')
       normalize = slemparams.normalize;
   else
       normalize = 1;
   end
   
   if isfield(slemparams, 'q_idx')
       q_idx = slemparams.q_idx;
       Nquery = size(q_idx, 2);
   else
       q_idx = 1:Nposi;
       Nquery = size(q_idx, 2);
   end
   
   if isfield(slemparams, 'theta')
       theta = slemparams.theta;
   else
       theta = 1;
   end

   if isfield(slemparams, 'useBdag')
       useBdag = slemparams.useBdag;
   else
       useBdag = 0;
   end
   
   if isfield(slemparams, 'rank_max')
       r = slemparams.rank_max;
       if isequal(kernel, 'rbf')
           kernel_id = 0;
       elseif isequal(kernel, 'linear')
           kernel_id = 1;
       elseif isequal(kernel, 'quad')
           kernel_id = 2;
       elseif isequal(kernel, 'poly')
           kernel_id = 3;
       end
   else
       r = n;
   end
   
   if isfield (slemparams, 'decomp_method')
       decomp_method = slemparams.decomp_method;
   else
       decomp_method = 'KPCA';
   end
   
   if isfield(slemparams, 'Nchunk')
       Nchunk = slemparams.Nchunk;
   else
       Nchunk = Nposi;
   end
   
   if isfield(slemparams, 'feature')
       feature = slemparams.feature;
   elseif p == 4096
       feature = 'netvlad';
   else
       error('Unknown base feature.')
   end
   
   
   if isfield(slemparams, 'database')
       database = slemparams.database;
   elseif Nposi == 1491
       database = 'holidays';
   elseif Nposi == 5063
       database = 'oxford';
   else
       database = '';
   end
   
   if ~exist(['./offline/' feature], 'dir')
       mkdir(['./offline/' feature])
   end
   if ~exist(['./offline/' feature '/' database], 'dir')
       mkdir(['./offline/' feature '/' database])
   end
end

normparams.kernel = 'linear';
normparams.gamma = 0;
            
if isequal(kernel, 'None')
    disp('SLEM: Square Loss Exemplar Machine');
    disp('Non-kernelized');
    disp('------------------------------------------');
    
    mu = mean(Xnega,2);
    A = 1/n*(Xnega*Xnega') - mu*mu' + lambda*eye(p);    
    delta  = Xposi-repmat(mu, 1, Nposi);% p\times Nposi
    %C = 2./(sum(delta.*(invA*delta))+1/theta+1);% \in 1\times Nposi
    
    Wposi = A\delta;
        
    if normalize
        for i=1:Nposi
            Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
        end
    else
        C = 2./(sum(delta.*(A\delta))+1/theta+1);% \in 1\times Nposi
        Wposi = Wposi.*repmat(C, p, 1);
    end
    
    simi = Wposi'*Wposi(:, q_idx);
    %write_mc(['/scratch/sampaiod/online-e-svm/mc_files/angle_score_' num2str(theta) '.mc'], Xposi, 1-simi)
else
    disp('SLEM: Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel]);
    disp('------------------------------------------');

    %% pre-processing Wnega
    t2 = tic;
    if ~ useBdag
        preprocessed_file = ['./offline/' feature '/' database ...
        '/preprocessed_slem_' kernel '_gamma_' num2str(gamma) '_n_' num2str(n) ...
        '_rank_' num2str(r) '.mat'];
    else
        preprocessed_file = ['./offline/' feature '/' database ...
        '/preprocessed_slem_' kernel '_gamma_' num2str(gamma) '_n_' num2str(n) ...
        '_rank_' num2str(r) '_Bdag.mat'];
    end
    if exist(preprocessed_file, 'file')
        load(preprocessed_file);
    elseif r == n
        K = kernel_matrix(Xnega', Xnega, slemparams);

        %ep = min(min(abs(K)))*10^-6;
        ep = 1e-4;
        B = chol(K+ep*eye(n))';
        
        
        mu = mean(B)';
        Sigma = 1/n*(B'*B) -mu*mu';
        if ~useBdag    
            save(preprocessed_file, 'B', 'Sigma', 'mu');
	else
	    Bdag = B\eye(n);
	    save(preprocessed_file, 'B', 'Sigma', 'mu', 'Bdag');
	end
    else
        switch decomp_method
            case 'KPCA'
                %% Kernel PCA
                K = kernel_matrix(Xnega', Xnega, slemparams);
                %[U,D] = svd(K);
                %B = U(:,1:r)*sqrt(D(1:r,1:r)); %n\times r
                [U,D] = eig(K);
                B = U(:,end+1-r:end)*sqrt(D(end+1-r:end,end+1-r:end));
            case 'ICD'
                %% Incomplete Cholesky 
                K = kernel_matrix(Xnega', Xnega, slemparams);
                [B, perm] = icd_general_m(K, 1e-8, r);
                Xnega = Xnega(:,perm);
                % to be verified
                %[B_0, perm] = incchol(Xnega, kernel_id, gamma, r);
                %[U,D] = svd(B_0*B_0'); 
                %B = U(:,1:r)*sqrt(D(1:r,1:r)); %n\times r
                %Xnega = Xnega(:,perm);
                %replace Xnega for Xnega(:,perm)
        end
        mu = mean(B)';
        Sigma = 1/n*(B'*B) -mu*mu';
        
        if ~useBdag     
            save(preprocessed_file, 'B', 'Sigma', 'mu');
	else
	    Bdag = B\eye(n);
	    save(preprocessed_file, 'B', 'Sigma', 'mu', 'Bdag');
	end
    end
    %lambda = lambda*(max(max(B)))^2;
    G = Sigma + lambda*eye(r);% r\times r
    t3 = toc(t2);
    disp(['pre-processing negative data time: ' num2str(t3)])
        
    %% processing Wposi
    t4 = tic;
    
    % a matrix n\times Nposi can be too big. We calculate by chunks
    steps = ceil(Nposi/Nchunk);
    
    k_00q = kernel_matrix(Xposi(:, q_idx)', Xposi, slemparams)';% Nposi\times Nquery
    %k_00q = k_00(:,q_idx);% Nposi\times Nquery
    if steps == 1
        if length(q_idx) == Nposi;
            k_00d = diag(k_00q);
        else
            k_00d  = kernel_diagonal(Xposi, slemparams);% Nposi\times 1
        end
        k_0 = kernel_matrix(Xnega', Xposi, slemparams);% n\times Nposi
	if ~ useBdag
            v = B\k_0; %r\times Nposi
	else
	    v = Bdag*k_0; %r\times Nposi
	end
        
        %% similarity calculation
        betahat = G\(v-repmat(mu, 1, size(v,2)));% r\times Nposi
        simi = betahat'*betahat(:,q_idx) + (k_00q-v'*v(:,q_idx))/lambda/lambda;
        if normalize
            if length(q_idx) == Nposi;
                NN = sqrt(diag(simi));
            else
                NN = sqrt(kernel_diagonal(betahat, normparams) + (k_00d-kernel_diagonal(v, normparams))/lambda/lambda);
            end
        end
        
        if exist('perm', 'var')
            v = B(1:r,:)\k_0(1:r,:); % r\times Nposi
            u = sqrt(k_00d' - kernel_diagonal(v, normparams)');% 1\times Nposi
    
            w = zeros(n, Nposi);
            w(r+1:n,:) = (1./repmat(u, n-r, 1)).*(k_0(r+1:n,:) - B(r+1:n,:)*v);
            wbar = mean(w);% 1\times Nposi
            
            a_00 = 1/n*kernel_diagonal(w, normparams)' - wbar.^2 + lambda;% 1\times Nposi
            a_0  = 1/n*B'*w - mu*wbar;% r\times Nposi
            
            eta = 1./(a_00 - diag(a_0'*(G\a_0))');% 1\times Nposi
            xi = u -wbar -diag(a_0'*(G\(v-repmat(mu,1,Nposi))))';% 1\times Nposi
            
            %beta_0 = eta.*xi;% 1\times Nposi
            %nu_kk = (theta-1)/(theta+1) -1/(theta+1)*diag(beta_kk'*([theta*u+wbar; theta*v+repmat(mu,1,Nposi)]))';
            betahat = -repmat(eta.*xi, r, 1).*(G\a_0)+G\(v-repmat(mu, 1, Nposi));% r \times Nposi
            simi = betahat'*betahat(:,q_idx) + (k_00q-v'*v(:,q_idx))/lambda/lambda;
            if normalize
                NN = sqrt(kernel_diagonal(betahat, normparams) + (k_00d-kernel_diagonal(v, normparams))/lambda/lambda);
            end
        end
    else
        %% TO BE REVISED
        % query specific matrices
        k_0q = kernel_matrix(Xnega', Xposi(:,q_idx), slemparams); % n\times Nquery
        vq = B\k_0q; % r\times Nquery
        betahatq = G\(vq-repmat(mu, 1, size(vq,2)));
    
        simi = zeros(Nposi, Nquery);
        if normalize
            NN = zeros(Nposi, 1);
        end
        
        for i = 1:steps
            fprintf('step %d/%d \n', i, steps);
            j = min(Nchunk*i, Nposi);
        
            k_00d  = kernel_diagonal(Xposi(:,Nchunk*(i-1)+1:j), slemparams);% Nposi\times 1
            k_0 = kernel_matrix(Xnega', Xposi(:,Nchunk*(i-1)+1:j), slemparams);% n\times Nchunk
            v = B\k_0; %r\times Nchunk
            %% similarity calculation
            betahat = G\(v-repmat(mu, 1, size(v,2)));
            simi(Nchunk*(i-1)+1:j,:) = betahat'*betahatq + (k_00q(Nchunk*(i-1)+1:j,:)-v'*vq)/lambda/lambda;
            if normalize
                NN(Nchunk*(i-1)+1:j) = sqrt(kernel_diagonal(betahat, normparams) + (k_00d-kernel_diagonal(v, normparams))/lambda/lambda );
            end
        end
    end
    
    %% similarity normalization
    if normalize
        simi = simi./(NN*NN(q_idx)');
    end
    
    t5 = toc(t4);
    disp(['processing positive data time: ' num2str(t5)])
end
t1 = toc(t0);
disp(['full run time: ' num2str(t1)])
if nargout > 1
    full_time = t1;
    varargout{1} = full_time;
    if nargout > 2
        nega_time = t3; posi_time = t5; 
        varargout{2} = nega_time; varargout{3} = posi_time; 
    end
end

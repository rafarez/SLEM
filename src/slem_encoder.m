function [betahat, v] = slem_encoder(Xposi, Xnega, slemparams)
t0 = tic;
[p, n] = size(Xnega);
Nposi = size(Xposi, 2);

defaut_params;
normparams.kernel = 'linear';
normparams.gamma = 0;
            
if isequal(kernel, 'None')
    disp('SLEM: Square Loss Exemplar Machine');
    disp('Non-kernelized');
    disp('----------------------------------------');
    
    mu = mean(Xnega,2);
    A = 1/n*(Xnega*Xnega') - mu*mu' + lambda*eye(p);    
    delta  = Xposi-repmat(mu, 1, Nposi);% p\times Nposi
    %C = 2./(sum(delta.*(invA*delta))+1/theta+1);% \in 1\times Nposi
    
    Omega = A\delta;
        
    if normalize
        for i=1:Nposi
            Omega(:,i) = Omega(:,i)/norm(Omega(:,i));
        end
    else
        C = 2./(sum(delta.*(A\delta))+1/theta+1);% \in 1\times Nposi
        Omega = Omega.*repmat(C, p, 1);
    end
    betahat = Omega;
    v = [];
else
    disp('SLEM: Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel]);
    disp('----------------------------------------');

    %% pre-processing Wnega
    t2 = tic;
    K = kernel_matrix(Xnega', Xnega, slemparams);

    ep = 1e-6;
    B = chol(K+ep*eye(n))';
        
    mu = mean(B)';
    G = 1/n*(B'*B) -mu*mu' +lambda*eye(n);
    
    t3 = toc(t2);
    disp(['pre-processing negative data time: ' num2str(t3)])
        
    %% processing Wposi
    t4 = tic;
    
    % a matrix n\times Nposi can be too big. We calculate by chunks
    steps = ceil(Nposi/Nchunk);
    
    %k_00q = kernel_matrix(Xposi(:, q_idx)', Xposi, slemparams)';% Nposi\times Nquery
    if steps == 1
        k_0 = kernel_matrix(Xnega', Xposi, slemparams);% n\times Nposi
        v = B\k_0; %r\times Nposi
        betahat = G\(v-repmat(mu, 1, size(v,2)));% n\times Nposi
            
       %% similarity calculation
        if ~normalize
            k_00d  = kernel_diagonal(Xposi, slemparams);% Nposi\times Nposi
            C = 2./(sum( (v-repmat(mu, 1, size(v,2))).*betahat ) +(k_00d-kernel_diagonal(v, normparams))'/lambda +1/theta+1);
        end
    else
        % query specific matrices
        k_0q = kernel_matrix(Xnega', Xposi(:,q_idx), slemparams); % n\times Nquery
        vq = B\k_0q; % r\times Nquery
        betahatq = G\(vq-repmat(mu, 1, size(vq,2)));
    
        simi = zeros(Nposi, Nquery);
        if normalize
            NN = zeros(Nposi, 1);
        end
        
        for i = 1:steps
            j = min(Nchunk*i, Nposi);
        
            k_00d  = kernel_diagonal(Xposi, slemparams);% Nposi\times Nposi
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
    if ~normalize
        v = v.*repmat(C, n, 1);
    end
    
    t5 = toc(t4);
    disp(['processing positive data time: ' num2str(t5)])
end
t1 = toc(t0);
disp(['full run time: ' num2str(t1)])
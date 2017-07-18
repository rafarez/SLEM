function simi = simple_full_encoder_oxford105(Xoxford, Xnega, Xflickr, query_list, kernel, gamma, lambda)
t0 = tic;
[p, n] = size(Xnega);
Nposi = size(Xoxford, 2);
Ndist = size(Xflickr, 2);
Nquery = size(query_list, 2);
XX = [Xoxford Xflickr];

kernel_type = {kernel, gamma};

if isequal(kernel, 'None')
    disp('SLEM: Square Loss Exemplar Machine');
    disp('Non-kernelized');
    disp('----------------------------------------');
    
    mu = mean(Xnega,2);
    
    A = 1/n*(Xnega*Xnega') - mu*mu' + lambda*eye(p);    
    delta = XX-repmat(mu, 1, Nposi+Ndist);% p\times Nposi
    
    Wposi = A\delta;
        
    %nu = (theta-1)/(theta+1) -1/(theta+1)*diag(Wposi'*(theta*Xposi+repmat(mu,1,Nposi)))';
    %Wposi = [nu; Wposi];
    for i=1:Nposi+Ndist
        Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
    end
    
    
    simi = Wposi'*Wposi(:, query_list);
else
    disp('SLEM: Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel_type{1}]);
    disp('----------------------------------------');

    %% pre-processing Wnega
    t2 = tic;
    K = kernel_matrix(Xnega', Xnega, kernel_type);

    ep = 1e-8;
    B = chol(K+ep*eye(n))';
        
    mu = mean(B)';
    G = 1/n*(B'*B) -mu*mu' +lambda*eye(n);
    
    t3 = toc(t2);
    disp(['pre-processing negative data time: ' num2str(t3)])
    
    %% processing Wposi
    t4 = tic;
    
    % query specific matrices
    k_0q = kernel_matrix(Xnega', Xoxford(:,query_list), kernel_type);
    vq = B\k_0q;
    
    k_00 = kernel_matrix(Xoxford(:, query_list)', XX, kernel_type)';
    
    % a matrix n\times (Nposi+Ndist) is too big. We calculate by chunks
    Nchunk = 5000;
    steps = ceil((Nposi+Ndist)/Nchunk);
    
    simi = zeros(Nposi+Ndist, Nquery);
    NN = zeros(Nquery,1);
    for i = 1:steps
        j = min(Nchunk*i, Nposi+Ndist)
        k_0 = kernel_matrix(Xnega', XX(:,5000*(i-1)+1:j), kernel_type);
        k_000 = kernel_matrix(XX(:,5000*(i-1)+1:j)', XX(:,5000*(i-1)+1:j), kernel_type);
        v = B\k_0;
        simi(5000*(i-1)+1:j,:) = (G\(v-repmat(mu, 1, size(v,2))))'*(G\(vq-repmat(mu, 1, size(vq,2)))) + (k_00(5000*(i-1)+1:j,:)-v'*vq)/lambda/lambda;
        NN(5000*(i-1)+1:j) = sqrt(diag((G\(v-repmat(mu, 1, size(v,2))))'*(G\(v-repmat(mu, 1, size(v,2)))) + (k_000-v'*v)/lambda/lambda ));
    end

    simi = simi./(NN*NN(query_list)');
    
    t5 = toc(t4);
    disp(['processing positive data time: ' num2str(t5)])
    end
t1 = toc(t0);
disp(['full run time: ' num2str(t1)])
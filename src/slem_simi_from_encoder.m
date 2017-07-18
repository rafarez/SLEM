function  simi = slem_simi_from_encoder(Xposi1, betahat1, v1, Xposi2, betahat2, v2, slemparams)

kernel = slemparams.kernel;
lambda = slemparams.lambda;

if isequal(kernel , 'None')
    simi = betahat1'*betahat2;
    else
    k_00q = kernel_matrix(Xposi(:, q_idx)', Xposi, slemparams)';
    
    simi = betahat'*betahat(:,q_idx) + (k_00q-v'*v(:,q_idx))/lambda/lambda;
    if normalize
        normparams.kernel = 'linear';
        normparams.gamma = 0;
        k_00d  = kernel_diagonal(Xposi, slemparams);
        
        NN = sqrt(kernel_diagonal(betahat, normparams) + (k_00d-kernel_diagonal(v, normparams))/lambda/lambda);
        simi = simi./(NN*NN(q_idx)');
    end
end

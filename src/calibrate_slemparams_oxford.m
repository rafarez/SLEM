function [params_star, mAP_star, mAP_tab] = calibrate_slemparams_oxford(Xposi, Xnega, slemparams, gamma_list, lambda_list, n_min, n_list, Y, indexes)

queries_name = {'all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', ...
    'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera'};
%% evaluation of simi
addpath('./eval_oxford/')

q_idx = slemparams.q_idx;

%% find opt. gamma and lambda in gamma_list and lambda_list, using n = n_min
mAP_star    = 0;
gamma_star  = 0;
lambda_star = 0;
mAP_tab = [];

for gamma = gamma_list
    slemparams.gamma = gamma;
    for lambda = lambda_list
        slemparams.lambda = lambda;
            
        simi = slem_similarity(Xposi, Xnega(:,1:n_min), slemparams);
        AP = zeros(1, 5*length(queries_name));
        for q = 1:length(queries_name)  
            for t = 1:5
                test_name = [queries_name{q} '_' num2str(t)];
                AP(1, 5*(q-1)+t) = compute_ap_oxford_105k(simi, Y, indexes, test_name, q_idx); %
            end
        end
        mAP = mean(AP)
        mAP_tab = [mAP_tab mAP];
        if mAP >= mAP_star
            sprintf('mAP = %0.4f, for gamma = %0.4e and lambda = %0.4e', mAP, gamma, lambda)
            mAP_star = mAP;
            gamma_star = gamma;
            lambda_star = lambda;
        end
    end
end
mAP_tab = reshape(mAP_tab, [numel(gamma_list), numel(lambda_list)]);
%% using best result for bigger values of n
disp('Testing for bigger values of n')
slemparams.gamma = gamma_star;
slemparams.lambda = lambda_star;
slemparams.n_opt = n_min;

for n = n_list
    simi = slem_similarity(Xposi, Xnega(:,1:n), slemparams);
    for q = 1:length(queries_name)  
        for t = 1:5
            test_name = [queries_name{q} '_' num2str(t)];
            AP(1, 5*(q-1)+t) = compute_ap_oxford_105k(simi, Y, indexes, test_name, q_idx); %
        end
    end
    mAP = mean(AP)
    
    if mAP > mAP_star
        sprintf('mAP = %0.4f, for n = %d', mAP, n)
        mAP_star = mAP;
        slemparams.n_opt = n;
    end
end

slemparams.mAP = mAP_star;
params_star = slemparams;
if 1
    save_slemparams(slemparams);
end
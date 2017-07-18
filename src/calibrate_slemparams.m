function calibrate_slemparams(Xposi, Xnega, slemparams, gamma_list, lambda_list, n_list)

savepath = '/scratch/sampaiod/online-e-svm/mc_files/';

if isfield(slemparams, 'rank_max')
    r = slemparams.rank_max;
    endfile = ['_r_'  num2str(r) '.mc'];
else
    endfile = '.mc';
end

for n = n_list
    for gamma = gamma_list
        slemparams.gamma = gamma;
        for lambda = lambda_list
            slemparams.lambda = lambda;
            
            simi = slem_similarity(Xposi, Xnega(:,1:n), slemparams);
            
            savefile = fullfile(savepath, sprintf('%s_%s_gamma_%s_lambda_%s_n_%s%s', ...
                slemparams.feature, slemparams.kernel, num2str(slemparams.gamma), ...
                num2str(slemparams.lambda), num2str(n), endfile));
            
            if 1
                disp(savefile)
            end
            write_mc(savefile, [], simi);
        end
    end
end
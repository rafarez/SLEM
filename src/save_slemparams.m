function save_slemparams(slemparams)
if ~isfield(slemparams, 'n_opt')
    disp('You should considere adding field "n_opt", to help to reproduce its results')
end

assert(isfield(slemparams, 'feature'), 'This structure does not have a "feature" field')
assert(isfield(slemparams, 'database'), 'This structure does not have a "database" field')

feature = slemparams.feature;
database = slemparams.database;

kernel = slemparams.kernel; 
if isequal(kernel, 'None') 
    kernel = 'linear';
end

if isfield(slemparams, 'rank_max')
    r = slemparams.rank_max;
    filename = ['params_' kernel '_rank_' num2str(r) '.mat'];
else
    filename = ['params_' kernel '.mat'];
end

if ~exist(fullfile(pwd, 'slemparams'), 'dir')
    mkdir(fullfile(pwd, 'slemparams'));
end
if ~exist(fullfile(pwd, 'slemparams', feature), 'dir')
    mkdir(fullfile(pwd, 'slemparams', feature));
end
if ~exist(fullfile(pwd, 'slemparams', feature, database), 'dir')
    mkdir(fullfile(pwd, 'slemparams', feature, database));
end
    
save( fullfile(pwd, 'slemparams', feature, database, filename), 'slemparams');


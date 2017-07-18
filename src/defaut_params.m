if true
   if isfield(slemparams, 'kernel')
       kernel = slemparams.kernel;
   else
       kernel = 'None';
   end
   
   if isfield(slemparams, 'gamma')
       gamma = slemparams.gamma;
   else
       gamma = 0.1;
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
   
   if isfield(slemparams, 'rank_max')
       r = slemparams.rank_max;
   else
       r = n;
   end
   
   if isfield(slemparams, 'Nchunk')
       Nchunk = slemparams.Nchunk;
   else
       Nchunk = Nposi;
   end
   
   if isfield(slemparams, 'feature')
       feature = slemparams.feature;
   elseif p == 512
       feature = 'spoc';
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
       if ~exist(['./offline/' feature '/' database], 'dir')
           mkdir(['./offline/' feature '/' database])
       end
   end
end
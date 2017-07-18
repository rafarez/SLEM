%% main.m: running code for 
% Rafael Rezende 27/09/2015

% setup path
addpath('./src')

% setup mc reader
mc_path = './matrix_chain/';
addpath(mc_path)

% setup incomplete Cholesky
cd ./cholesky/
mex incchol.c
cd ..
addpath('./cholesky/');

% setup APT
apt_path = '/scratch/sampaiod/APT.1.4/';
addpath(apt_path)
APT_compile

experiment = 'real data';
switch experiment
    case 'toy'
        % experiment_type = 1: 'good' negative data. experiment = 2: 'bad' negative data.
        experiment_type = 1;  

        % makes sure different runs of your code will have same data 
        randn('seed',0);

        [Xposi,Xnega] = generate_positive_negative(100,experiment_type);
        
        % show your data
        figure;
        scatter(Xposi(1,:),Xposi(2,:),'g','filled'); hold on;
        scatter(Xnega(1,:),Xnega(2,:),'r','filled'); 
       
    case 'real data'
        %database = 'holidays';
        database = 'oxford';
        features = 'netvlad_ox_nopca';%netvlad_hol
        %features = 'vlad';
        %features = 'cnn';
        normalized = 1;
        generic_negative_data = 14500;
        %[Xposi, Xnega, Y] = load_mc_data('holidays', features, normalized, generic_negative_data);
        [Xposi, Xnega, Y, indexes] = load_mc_data(database, features, normalized, generic_negative_data);
end

%% getting ride of the NaN from Xnega (cnn features)
Sum = sum(Xnega);
good_idx = Sum<Inf;
Xnega = Xnega(:,good_idx);

%% permutating indexes of Xposi (for holidays)
P = zeros(size(Y,2),1);
q_idx = zeros(size(Y,1), 1);
count = 0;
for i=1:size(Y,1)
    idx = find(Y(i,:)==1);
    l = length(idx);
    P(count+1:count+l) = idx;
    q_idx(i) = count+1;
    count = count+l;
end
Xposi = Xposi(:,P);

%% compressing Xposi and Xnega (for netvlad)
D = 512;
Xposi = Xposi(1:D,:);
Xnega = Xnega(1:D,:);
for i = 1:size(Xposi,2)
    Xposi(:,i) = Xposi(:,i)/norm(Xposi(:,i));
end
for i = 1:size(Xnega,2)
    Xnega(:,i) = Xnega(:,i)/norm(Xnega(:,i));
end

%% list of queries to evaluate (for oxford)
queries_name = {'all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket', ...
    'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera'};
%% evaluation of simi
addpath('./eval_oxford/')

slemparams.kernel    = 'quad';
slemparams.gamma     = 0.01;
slemparams.lambda    = 10^-3;
slemparams.normalize = true;
slemparams.q_idx = q_idx;

simi = slem_similarity(Xposi, Xnega(:,1:10000), slemparams);

q_idx = zeros(1, 5*length(queries_name)); 

t0 = tic;
AP = zeros(1, 5*length(queries_name));
for q = 1:length(queries_name)  
    for t = 1:5
        test_name = [queries_name{q} '_' num2str(t)];
        %disp(test_name)
        %AP(1, 5*(q-1)+t) = compute_ap_oxford(simi, Y, indexes, test_name);
        [AP(1, 5*(q-1)+t) q_idx(5*(q-1)+t)] = compute_ap_oxford(simi, Y, indexes, test_name);
        %AP(1, 5*(q-1)+t) = compute_ap_oxford_105k(simi, Y, indexes, test_name, q_idx); %
    end
end
mAP = mean(AP)
t1 = toc(t0)
% 
% results_path = './eval_oxford/baseline/';
% for q = 1:length(queries_list)
%     for t = 1:5
%         test_name = [queries_list{q} '_' num2str(t)];
%         disp(test_name)
%         write_lists_oxford(simi, Y, indexes, test_name, results_path);
%     end
% end

%% eliminate NaNs from Xnega
Sum = sum(Xnega);
good_idx = find(Sum<Inf);
Xnega = Xnega(:,good_idx);

[p, n] = size(Xnega);
% full encoder base parameters
k=1;
theta = 1/30;
lambda = 10^-3;
w_normalized = 1;

%% parameter to validate in parallel
lambda = [.001, .003, .01, .03, .1, 1];
theta = [10, 10/3, 1, 1/3, 1/10];

%% choose your parallel run
%APT_compile
%simi = APT_run('simple_full_encoder', {Xposi}, {Xnega}, {'poly'}, gamma_list, theta, lambda_list, 1, 'Memory', 10000, 'CombineArgs', 1);
for i = 1:length(gamma_list)
    for j = 1:length(lambda_list)
        disp(['gamma = ' num2str(gamma_list(i)) ', lambda = ' num2str(lambda_list(j))])
        AP = zeros(1, 5*length(queries_name));
        for q = 1:length(queries_name)
            for t = 1:5
                test_name = [queries_name{q} '_' num2str(t)];
                %disp(test_name)
                AP(1, 5*(q-1)+t) = compute_ap_oxford(simi, Y, indexes, test_name);
            end
        end
        mAP = mean(AP)
    end
end


[Wposi_f, simi_f] = APT_run('quick_full_encoder', {Xposi}, {Xnega}, {kernel_type}, {k}, theta, lambda, {w_normalized}, 'Memory', 50000, 'CombineArgs', 1);
for i = 1:(length(theta)*length(lambda))
    disp(['lambda = ' num2str(para_lambda{i}) ', theta = ' num2str(para_theta{i})])
    disp(simi_b{i}(1:4,1:4))
    disp(simi_f{i}(1:4,1:4))
    filename = ['./mc_files/basic_None_lambda_' num2str(para_lambda{i}) '_theta_' num2str(para_theta{i}) '.mc'];
    write_mc(filename, Wposi_b{i}, simi_b{i}, para_theta{i}, para_lambda{i});
    filename = ['./mc_files/full_None_lambda_' num2str(para_lambda{i}) '_theta_' num2str(para_theta{i}) '.mc'];
    write_mc(filename, Wposi_f{i}, simi_f{i}, para_theta{i}, para_lambda{i});
end

for j=1:length(lambda)
    Wposi = pre_Wposi{j}(:,P);
    simi = pre_simi{j}(P,P);
end
filename = './mc_files/linear_lambda_e-2_r_2048.mc';
%out = read_mc(filename);
%pre_simi = out{2};
%pre_Wposi = out{1};
Wposi = pre_Wposi(:,P);
simi = pre_simi(P,P);
%write_mc(filename, Wposi, simi, out{3}, out{4}, out{5});
write_mc(filename, Wposi, simi, out{3}, out{4});

%% test different ranks for linear kernel
kernel_type = {'linear'};
theta = 10/3;
lambda = 0.003;
rank_max = [8192 4096 2048 1024 512 256 128 64];

[Wposi, simi] = APT_run('quick_full_encoder', {Xposi}, {Xnega}, {kernel_type}, {k}, theta, lambda, {w_normalized}, rank_max, 'Memory', 4000);
for i=1:length(rank_max)
    r = rank_max(i);
    filename = ['./mc_files/full_linear_r_' num2str(r) '.mc']
    write_mc(filename, Wposi{i}, real(simi{i}), r)
    disp(simi{i}(1:10,1:10))
end

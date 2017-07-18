distrac_path = './vlad-64/bigimbaz/flickrtar/tmi_data/pyrBldr_4-3,6-3,8-3,10-3/vlad-64/descriptors/';
addpath(distrac_path)

distrac_dirs = dir(fullfile(distrac_path));

d = 600;
Xdist12 = zeros(p, 50*1000);
count = 0;
for dd = 551:d
    disp(dd)
    dir_info = dir(fullfile([distrac_path  distrac_dirs(dd+2).name], '*.mc'));
    for i = 1:length(dir_info)
        filedist = [distrac_path '/' distrac_dirs(dd+2).name '/' dir_info(i).name];
        out = read_mc(filedist);
        if ~isempty(out)
            count = count + 1;
            Xdist12(:,count) = out{1}';
        end
    end
end

if (count<50*1000)
    Xdist12 = Xdist12(:,1:count);
end
save('/meleze/data1/sampaiod/SLEM_largescale/vlad_distractors12.mat', 'Xdist12', '-v7.3')
clear Xdist12


simi = getThoseSimi2(Xposi, Xdist12, Xnega, G, mu, beta_hatP, kernel_type, B, lambda, vP, simiP);
AP = zeros(1, 5*length(queries_list));
query_idx = zeros(1, 5*length(queries_list));
for q = 1:length(queries_list)
    for t = 1:5
        test_name = [queries_list{q} '_' num2str(t)];
        %disp(test_name)
        [AP(1, 5*(q-1)+t) query_idx(1, 5*(q-1)+t)] = compute_ap_oxford(simi, Y, indexes, test_name);
    end
end
mAP = mean(AP)


filename = ['./mc_files/vlad_poly_dist_' num2str(d*1000) '.mc']
write_mc(filename, [], simi);

simi1 = Xposi'*[Xposi Xdist12];

filename = ['./mc_files/vlad_test_dist_' num2str(d*1000) '.mc']
write_mc(filename, Xposi, simi1)

deltaD = [Xposi Xdist12] - repmat(mu, 1, Nposi+count);
WD = A\deltaD;

if beta_normalize
    for i = 1:(Nposi+count)
        WD(:,i) = WD(:,i)/norm(WD(:,i));
    end
end

simi = Wposi'*WD;

filename = ['./mc_files/vlad_None_dist_' num2str(d*1000) '.mc']
write_mc(filename, Xposi, simi)


[simi, simi1] = APT_run('getThoseSimi.m', {Xposi}, {Xdist12}, {A}, {mu}, {Wposi}, 'Memory', 40000);

filename = ['./mc_files/vlad_test_dist_' num2str(d*1000) '.mc']
write_mc(filename, [], simi1{1})

filename = ['./mc_files/vlad_None_dist_' num2str(d*1000) '.mc']
write_mc(filename, [], simi{1})


%% for oxford
Xnega1 = Xnega(:,1:10000);
Xnega2 = Xnega(:,10001:20000);
kernel_type = {'poly', 0.03};
lambda = 10^-4.5;
gamma = 0.03;


%% for oxford 105k
    for i = 1:(size(X1, 2))
        t0 = tic;
        NN = (G\(v(:,i)-mu))'*(G\(v(:,i)-mu)) + (kernel_matrix(X1(:,i)', X1(:,i), kernel_type) - v(:,i)'*v(:,i))/lambda/lambda;
        simi(i,:) = simi(i,:)/sqrt(NN);
        t1 = toc(t0)
        
    end
    
    for i = 1:(size(X3, 2))
        NN = (G\(v(:,i+Nposi)-mu))'*(G\(v(:,i+Nposi)-mu)) + (kernel_matrix(X3(:,i)', X3(:,i), kernel_type) - v(:,i+Nposi)'*v(:,i+Nposi))/lambda/lambda;
        simi(i+Nposi,:) = simi(i+Nposi,:)/sqrt(NN);
    end
    
    for i = 1:length(query_idx)
        j = query_idx(i);
        NN = (G\(v(:,j)-mu))'*(G\(v(:,j)-mu)) + (kernel_matrix(X1(:,j)', X1(:,j), kernel_type) - v(:,j)'*v(:,j))/lambda/lambda;
        simi(:,i) = simi(:,i)/sqrt(NN);
    end


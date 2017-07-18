%% VLAD for image search
% to be corrected
% setup VL FEAT
addpath('/scratch/sampaiod/toolbox/vlfeat-0.9.18/toolbox')
vl_setup

% path to database
db1_dir = '/scratch/sampaiod/Database/Holidays/';
db2_dir = '/sequoia/data1/sampaiod/VOC2007/VOCdevkit/JPEGImages/';

db_nega_dir = {};
db_nega_dir{1} = '/scratch/sampaiod/Database/Flickr100K/oxc1_100k/california/';
db_nega_dir{2} = '/scratch/sampaiod/Database/Flickr100K/oxc1_100k/landscape/';
db_nega_dir{3} = '/scratch/sampaiod/Database/Flickr100K/oxc1_100k/ocean/';
db_nega_dir{4} = '/scratch/sampaiod/Database/Flickr100K/oxc1_100k (2)/holiday/';
db_nega_dir{5} = '/scratch/sampaiod/Database/Flickr100K/oxc1_100k (2)/vacation/';
nega_len = zeros(size(db_nega_dir));

sizeStep = 4;
sizeBin = 24;
scalesList = 2.^[-1:.5:1];
newSize = [316 316];
%% learn center of clusters on VOC2007

VOC_info = dir(fullfile(db2_dir, '*.jpg'));

Flickr_info = [];
nega_count = 0;
for i = 1:length(db_nega_dir)
    Flickr_info = [Flickr_info; dir(fullfile(db_nega_dir{i}, '*.jpg'))];
    nega_len(i) = length(Flickr_info);
end

num_sample_images = 100;

allSift = [];
for i = 1:num_sample_images
    disp(i)
    imname = [db_nega_dir{1} Flickr_info(i).name];
    im = imresize(imread(imname), newSize);
    for scale = scalesList
        [~, denseSift] = vl_dsift(rgb2gray(im2single(imresize(im, scale))), 'size', sizeBin, 'step', sizeStep);
        allSift = [allSift denseSift];
    end
end
clear denseSift
allSift = single(allSift);
% SIFT -> RootSIFT
ep = 10^-4;
for j = 1:size(allSift, 2)
    %disp(j)
    allSift(:,j) = sqrt(allSift(:,j)/max(ep, norm(allSift(:,j))) ); % l_2 normalization
    %allSift(:,j) = sqrt(allSift(:,j)/max(ep, sum(allSift(:,j))) ); % l_1 normalization
end

% applying PCA
%covMatrix = cov(allSift');%single(allSift)*single(allSift');
%[eig_vectors, eig_values] = pcacov(covMatrix);

% PCA reduction
%D = 64;
%W = eig_vectors(:,1:D);
%reducedDimData = W'*single(allSift);

% applying K-means
K = 2^6;
[mu, label] = vl_kmeans(allSift, K); %D \times K

% LCS: applying a PCA for each codeword
W = cell(K,1);
D = 128;
for k = 1:K
    sk_idx = find(label==k);
    disp([k, length(sk_idx)])
    sk = allSift(:,sk_idx);
    covMatrix = cov(sk');%sk*sk';
    W{k} = pcacov(covMatrix);
end
    

%% loading Holidays db
holidays_info = dir(fullfile(db1_dir, '*.jpg'));
num_images = length(holidays_info);

enc_final = zeros(5*D*K, num_images);
%Xposi = zeros(D*K, num_images);
%% holidays' enc
for i = 1:num_images
    t0 = tic;
    disp(i)
    count = 1;
    imname = [db1_dir holidays_info(i).name];
    featname = ['./local_descriptors/holidays/RootSIFT_VLAD/' holidays_info(i).name(1:end-4) '_l2norm.mat'];
    if exist(featname, 'file')
        load(featname);
    else
        im = imresize(imread(imname), newSize);
        X = []; % 128 \times N
        frames = [];
        for scale = scalesList
            [fr, denseSift] = vl_dsift(rgb2gray(im2single(imresize(im, scale))), 'size', sizeBin, 'step', sizeStep);
            X = [X denseSift];
            fr = (fr - repmat(scale*newSize'/2, 1, size(fr,2)))./(repmat(scale*newSize'/2, 1, size(fr,2)));
            frames = [frames fr];
        end
        X = single(X);
        ep = 10^-4;
        for j = 1:size(X, 2)
            %disp(j)
            X(:,j) = sqrt(X(:,j)/max(ep, norm(X(:,j))) ); %l_2 normalization
            %X(:,j) = sqrt(X(:,j)/max(ep, sum(X(:,j))) ); %l_1 normalization
        end
        %% finding assigments
        N = size(X, 2);
        %Q = single(zeros(K, N));
        
        q = single(zeros(1, N));
        dist = zeros(K, 1);   
        for n = 1:N
            %t0 = tic;
            tt = repmat(X(:,n), 1, K) - mu;
            dist = diag(tt'*tt);
            q(n) = find(dist == min(dist), 1);
            %Q(q, n) = 1;
            %t1(n) = toc(t0);
        end
        %t1 = toc(t0)
        
        save(featname, 'X', 'frames', 'q', 'mu')
    end
    %t1 = toc(t0)
    %% pyramid level 1
    %count = 1;
    %t2 = tic;
    enc = zeros(D*K, 1);
    for k = 1:K
        %s_idx = find(Q(k,:) == 1);
        s_idx = find(q == k);
        
        if isempty(s_idx)
            enc(D*(k-1)+1:D*k, :) = zeros(D, 1);
        else
            % points of the k-th cluster
            s = X(:, s_idx);
            % distances
            % dists = s - repmat(mu(:,k), 1, length(s));
        
            % distances normalized
            %dists = zeros(size(s));
            v_k = zeros(D, 1);
            M = mu(:,k);
            for j = 1:size(s_idx, 2)
                %dists(:,j) = (s(:,j)-mu(:,k))/max(norm(s(:,j)-mu(:,k)), ep);
                v_k = v_k + W{k}'*(s(:,j)-M)/max(norm(s(:,j)-M), ep);
            end
        
            %enc(D*(k-1)+1:D*k, :) = sum(dists, 2);%
            enc(D*(k-1)+1:D*k, :) = v_k;
            
        end
    end
    %t3 = toc(t2)
    alpha = .2;
    enc = sign(enc).*(abs(enc).^alpha);
    %enc = enc/norm(enc);
    %Xposi(D*K*(count-1)+1:D*K*count, i) = enc;
    enc_final(D*K*(count-1)+1:D*K*count, i) = enc;
    count = count + 1;
    %% pyramid level 2
    quad = cell(2,2);
    for j = 1:2
        bol1 = 2*(j>1)-1;
        x_frames = find(bol1*frames(1,:) < 0);
        for jj = 1:2
            bol2 = 2*(jj>1)-1;
            y_frames = find(bol2*frames(2,:) < 0);
            quad_idx = intersect(x_frames, y_frames);
            quad_X = X(:, quad_idx);
            quad_q = q(quad_idx);
            enc = zeros(D*K, 1);
            for k = 1:K
                %s_idx = find(Q(k,:) == 1);
                s_idx = find(quad_q == k);
        
                if isempty(s_idx)
                    enc(D*(k-1)+1:D*k, :) = zeros(D, 1);
                else
                    % points of the k-th cluster
                    s = quad_X(:, s_idx);
            
                    % distances normalized
                    %dists = zeros(size(s));
                    v_k = zeros(D, 1);
                    M = mu(:,k);
                    for j = 1:size(s_idx, 2)
                        %dists(:,j) = (s(:,1)-mu2(:,k))/max(norm(s(:,1)-mu2(:,k)), ep);
                        v_k = v_k + W{k}'*(s(:,j)-M)/max(norm(s(:,j)-M), ep);
                    end
        
                    enc(D*(k-1)+1:D*k, :) = v_k;
                end
            end
            enc = sign(enc).*(abs(enc).^alpha);
            enc = enc/norm(enc);
            enc_final(D*K*(count-1)+1:D*K*count, i) = enc;
            count = count+1;
        end
    end
    %enc_final = enc_final/norm(enc_final);
    t1 = toc(t0)
end

num_images_flickr = length(Flickr_info);
nega_dir_idx = 1;

enc_final = zeros(5*D*K, num_images_flickr);
%Xposi = zeros(D*K, num_images);
%% Flickr's enc
for i = 1:num_images_flickr
%for i = num_images +1:num_images_flickr
    t0 = tic;
    disp(i)
    if i>nega_len(nega_dir_idx)
        nega_dir_idx = nega_dir_idx+1;
    end
    count = 1;
    imname = [db_nega_dir{nega_dir_idx} Flickr_info(i).name];
    featname = ['./local_descriptors/Flickr/RootSIFT_VLAD/' Flickr_info(i).name(1:end-4) '_l2norm.mat'];
    if exist(featname, 'file')
        load(featname);
    else
        im = imresize(imread(imname), newSize);
        X = []; % 128 \times N
        frames = [];
        for scale = scalesList
            [fr, denseSift] = vl_dsift(rgb2gray(im2single(imresize(im, scale))), 'size', sizeBin, 'step', sizeStep);
            X = [X denseSift];
            fr = (fr - repmat(scale*newSize'/2, 1, size(fr,2)))./(repmat(scale*newSize'/2, 1, size(fr,2)));
            frames = [frames fr];
        end
        X = single(X);
        ep = 10^-4;
        for j = 1:size(X, 2)
            %disp(j)
            X(:,j) = sqrt(X(:,j)/max(ep, norm(X(:,j))) ); %l_2 normalization
            %X(:,j) = sqrt(X(:,j)/max(ep, sum(X(:,j))) ); %l_1 normalization
        end
        %% finding assigments
        N = size(X, 2);
        %Q = single(zeros(K, N));
        
        q = single(zeros(1, N));
        dist = zeros(K, 1);   
        for n = 1:N
            %t0 = tic;
            tt = repmat(X(:,n), 1, K) - mu;
            dist = diag(tt'*tt);
            q(n) = find(dist == min(dist), 1);
            %Q(q, n) = 1;
            %t1(n) = toc(t0);
        end
        %t1 = toc(t0)
        
        save(featname, 'X', 'frames', 'q', 'mu')
    end
    %t1 = toc(t0)
    %% pyramid level 1
    %count = 1;
    %t2 = tic;
    enc = zeros(D*K, 1);
    for k = 1:K
        %s_idx = find(Q(k,:) == 1);
        s_idx = find(q == k);
        
        if isempty(s_idx)
            enc(D*(k-1)+1:D*k, :) = zeros(D, 1);
        else
            % points of the k-th cluster
            s = X(:, s_idx);
            % distances
            % dists = s - repmat(mu(:,k), 1, length(s));
        
            % distances normalized
            %dists = zeros(size(s));
            v_k = zeros(D, 1);
            M = mu(:,k);
            for j = 1:size(s_idx, 2)
                %dists(:,j) = (s(:,j)-mu(:,k))/max(norm(s(:,j)-mu(:,k)), ep);
                v_k = v_k + W{k}'*(s(:,j)-M)/max(norm(s(:,j)-M), ep);
            end
        
            %enc(D*(k-1)+1:D*k, :) = sum(dists, 2);%
            enc(D*(k-1)+1:D*k, :) = v_k;
            
        end
    end
    %t3 = toc(t2)
    alpha = .2;
    enc = sign(enc).*(abs(enc).^alpha);
    %enc = enc/norm(enc);
    %Xposi(D*K*(count-1)+1:D*K*count, i) = enc;
    enc_final(D*K*(count-1)+1:D*K*count, i) = enc;
    count = count + 1;
    %% pyramid level 2
    quad = cell(2,2);
    for j = 1:2
        bol1 = 2*(j>1)-1;
        x_frames = find(bol1*frames(1,:) < 0);
        for jj = 1:2
            bol2 = 2*(jj>1)-1;
            y_frames = find(bol2*frames(2,:) < 0);
            quad_idx = intersect(x_frames, y_frames);
            quad_X = X(:, quad_idx);
            quad_q = q(quad_idx);
            enc = zeros(D*K, 1);
            for k = 1:K
                %s_idx = find(Q(k,:) == 1);
                s_idx = find(quad_q == k);
        
                if isempty(s_idx)
                    enc(D*(k-1)+1:D*k, :) = zeros(D, 1);
                else
                    % points of the k-th cluster
                    s = quad_X(:, s_idx);
            
                    % distances normalized
                    %dists = zeros(size(s));
                    v_k = zeros(D, 1);
                    M = mu(:,k);
                    for j = 1:size(s_idx, 2)
                        %dists(:,j) = (s(:,1)-mu2(:,k))/max(norm(s(:,1)-mu2(:,k)), ep);
                        v_k = v_k + W{k}'*(s(:,j)-M)/max(norm(s(:,j)-M), ep);
                    end
        
                    enc(D*(k-1)+1:D*k, :) = v_k;
                end
            end
            enc = sign(enc).*(abs(enc).^alpha);
            enc = enc/norm(enc);
            enc_final(D*K*(count-1)+1:D*K*count, i) = enc;
            count = count+1;
        end
    end
    %enc_final = enc_final/norm(enc_final);
    t1 = toc(t0)
end


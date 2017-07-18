function [Xposi, Xnega, Y, indexes] = load_mc_data(database, features, ...
    normalized, generic_negative_data)

home_path = cd;
db_path = home_path;

if isequal(features, 'cnn')
    db_path = [db_path '/caffe_descs']; 
    descriptor_size = 4096;

    if (generic_negative_data>0)
        pre_nega_path = [db_path '/bigimbaz/Flickr60K/tmi_data'];
        if normalized
            pre_nega_path = [pre_nega_path '/caffe_ilsvrc_ov0_fc6_l2norm/descriptors'];
        else
            pre_nega_path = [pre_nega_path '/caffe_ilsvrc_ov0_fc6/descriptors'];
        end
        Flickr_folders = {'/2270','/2271','/2272','/2273','/2274','/2275',...
            '/2276','/2277','/2278','/2279','/2280','/2281'};
        Flickr_sizes = [6221 6005 1934 6162 6128 6045 6109 6140 6195 ...
            6225 6003 6076];
        Xnega = zeros(descriptor_size,sum(Flickr_sizes(1:generic_negative_data)));
    end
    
    if isequal(database, 'holidays')
        db_path = [db_path '/holidays/tmi_data'];
        if normalized
            db_path = [db_path '/caffe_ilsvrc_ov0_fc6_l2norm/descriptors'];
        else
            db_path = [db_path '/caffe_ilsvrc_ov0_fc6/descriptors'];
        end
    elseif isequal(database, 'oxford')
        db_path = [db_path '/oxford/tmi_data/caffe_ilsvrc_ov0_fc6_l2norm/descriptors'];
    elseif isequal(database, 'VOC2007')
        db_path = [db_path '/VOC2007/tmi_data/caffe_ilsvrc_ov0_fc6_l2norm/descriptors/train'];
        %error('database to be done');
    end
    
elseif isequal(features, 'vlad')
    db_path = [db_path '/vlad-64']; 
    descriptor_size = 8192; 
    
    if (generic_negative_data>0)
        pre_nega_path = [db_path '/bigimbaz/Flickr60K/tmi_data/pyrBldr_4-3,6-3,8-3,10-3/vlad-64/descriptors'];
        Flickr_folders = {'/2270','/2271','/2272','/2273','/2274','/2275',...
            '/2276','/2277','/2278','/2279','/2280','/2281'};
        Flickr_sizes = [6218 6004 1929 6161 6128 6044 6109 6140 6195 ...
            6225 6003 6076];
        Xnega = zeros(descriptor_size,sum(Flickr_sizes(1:generic_negative_data)));
    end

    if isequal(database, 'holidays')
        db_path = [db_path '/holidays/tmi_data/pyrBldr_4-3,6-3,8-3,10-3/vlad-64/descriptors'];
    elseif isequal(database, 'oxford')
        db_path = [db_path '/oxford/tmi_data/pyrBldr_4-3,6-3,8-3,10-3/vlad-64/descriptors'];
    elseif isequal(database, 'VOC2007')
        %db_path = [db_path '/VOC2007/tmi_data/caffe_ilsvrc_ov0_fc6_l2norm/descriptors/train'];
        error(['database to be done for ' features ' features']);
    end
  
elseif isequal(features, 'hist')
    db_path = '/scratch/sampaiod/SpatialPyramid/'; 
    descriptor_size = 200;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr_features/'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end
    
    db_path = [db_path database '_features/'];
elseif isequal(features, 'pyramid')
    db_path = '/scratch/sampaiod/SpatialPyramid/'; 
    descriptor_size = 4200;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr_features/'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end
    
    db_path = [db_path database '_features/'];
elseif isequal(features, 'netvlad_1')
    db_path = '/scratch/sampaiod/online-e-svm/netvlad/';
    descriptor_size = 4096;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr/vd16_tokyoTM_conv5_3_preL2_intra_white'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
	db_path = [db_path 'holidays_rot/vd16_tokyoTM_conv5_3_preL2_intra_white/'];
    elseif isequal(database, 'oxford')
	db_path = [db_path 'oxford/vd16_tokyoTM_conv5_3_preL2_intra_white/'];
    end
elseif isequal(features, 'netvlad_hol')
    db_path = '/scratch/sampaiod/online-e-svm/netvlad/';
    descriptor_size = 4096;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra_white'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
	db_path = [db_path 'holidays_rot/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra_white/'];
    elseif isequal(database, 'oxford')
	db_path = [db_path 'oxford/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra_white/'];
    end
elseif isequal(features, 'netvlad_hol_nopca')
    db_path = '/scratch/sampaiod/online-e-svm/netvlad/';
    descriptor_size = 4096;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
	db_path = [db_path 'holidays_rot/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra/'];
    elseif isequal(database, 'oxford')
	db_path = [db_path 'oxford/vd16_offtheshelf_conv5_3_pitts30k_train_vlad_preL2_intra/'];
    end
elseif isequal(features, 'netvlad_ox')
    db_path = '/scratch/sampaiod/online-e-svm/netvlad/';
    descriptor_size = 4096;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr/vd16_pitts30k_conv5_3_vlad_preL2_intra_white_2'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
	db_path = [db_path 'holidays_rot/vd16_pitts30k_conv5_3_vlad_preL2_intra_white_2/'];
    elseif isequal(database, 'oxford')
	db_path = [db_path 'oxford/vd16_pitts30k_conv5_3_vlad_preL2_intra_white_2/'];
    end
elseif isequal(features, 'netvlad_ox_nopca')
    db_path = '/scratch/sampaiod/online-e-svm/netvlad/';
    descriptor_size = 4096;

    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr/vd16_pitts30k_conv5_3_vlad_preL2_intra'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
	db_path = [db_path 'holidays_rot/vd16_pitts30k_conv5_3_vlad_preL2_intra/'];
    elseif isequal(database, 'oxford')
	db_path = [db_path 'oxford/vd16_pitts30k_conv5_3_vlad_preL2_intra/'];
    end
elseif isequal(features, 'mac')
    db_path = '/scratch/sampaiod/online-e-svm/siaMAC/';
    descriptor_size = 512;
    
    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr10k/mac'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
        db_path = [db_path 'holidays_rot/mac/'];
    elseif isequal(database, 'oxford')
        db_path = [db_path 'oxford/mac/'];
    end
elseif isequal(features, 'rmac')
   db_path = '/scratch/sampaiod/online-e-svm/siaMAC/';
    descriptor_size = 512;
    
    if (generic_negative_data>0)
        db_nega_path = [db_path 'Flickr10k/rmac'];
        Flickr_folders = {};
        Xnega = zeros(descriptor_size, generic_negative_data);
    end

    if isequal(database, 'holidays')
        db_path = [db_path 'holidays_rot/rmac/'];
    elseif isequal(database, 'oxford')
        db_path = [db_path 'oxford/rmac/'];
    end 
else
    error('Wrong argument "features"')
end


addpath(db_path)
content = ls(db_path);
file_idx = strfind(content, '.mc');
if isequal(features, 'hist')
    db_info = dir(fullfile(db_path, '*hist_200.mat'));
elseif isequal(features, 'pyramid')
    db_info = dir(fullfile(db_path, '*_multi_pyramid_200_3.mat'));
else %if isequal(features, 'netvlad_1')
    db_info = dir(fullfile(db_path, '*.mc'));
end
N = length(db_info);


%% this part depend on the length of the name of the files' names
if isequal(database, 'holidays')
    nqueries = 500;
    Xposi = zeros(descriptor_size, N);
    Y = zeros(nqueries, N);
    indexes = NaN;
    for i = 1:N
        %filename = content(file_idx(i)-6:file_idx(i)+2);
        filename = db_info(i).name;
        if isequal(features, 'hist')
            out = load(filename);
            Xposi(:,i) = out.H;
        elseif isequal(features, 'pyramid')
            out = load(filename);
            Xposi(:,i) = out.pyramid;
        else
            out = read_mc(filename);
            Xposi(:,i) = out{1}';
        end
        %disp(i)
        %label = str2double(content(file_idx(i)-5:file_idx(i)-3))+1;
        label = str2double(filename(2:4))+1;
        Y(label, i) = 1;
    end
elseif isequal(database, 'oxford')
    Xposi = zeros(descriptor_size, N);
    Y = content;
    indexes = zeros(1, 2*N);
    indexes(2:2:2*N) = file_idx -1;
    space_idx = [0 find(isspace(content))];
    NN = length(space_idx)-1;
    im_count = 0;
%    q_count = 0;
    for j= 1:NN
        filename = content(space_idx(j)+1:space_idx(j+1)-1);
        if length(filename) > 1
            im_count = im_count+1;
            indexes(2*im_count-1) = space_idx(j)+1;
            out = read_mc(filename);
            if ~isempty(out)
                Xposi(:,im_count) = out{1}';
            end
        end
    end
% elseif isequal(database, 'hist')
%     Xposi = zeros(descriptor_size, N);
%     for i = 1:N
%         filename = db_info(i).name;
%         out = load(filename);
%         Xposi(:,i) = out.pyramid';
%         %disp(i)
%     end
% elseif isequal(database, 'pyramid')
%     Xposi = zeros(descriptor_size, N);
%     for i = 1:N
%         filename = db_info(i).name;
%         out = load(filename);
%         Xposi(:,i) = out.pyramid';
%         %disp(i)
%     end
elseif isequal(database, 'VOC2007')
    Xposi = zeros(descriptor_size, N);
    for i = 1:N
        filename = content(file_idx(i)-6:file_idx(i)+2);
        out = read_mc(filename);
        Xposi(:,i) = out{1}';
        %disp(i)
    end
end

count = 0;
if isempty(Flickr_folders)
    addpath(db_nega_path)
    if isequal(features, 'hist')
        nega_info = dir(fullfile(db_nega_path, '*hist_200.mat'));
        
        for i = 1:generic_negative_data
            filename = nega_info(i).name;
            out = load(filename);
            Xnega(:,i) = out.H;
        end
    elseif isequal(features, 'pyramid')
        nega_info = dir(fullfile(db_nega_path, '*pyramid_200_3.mat'));
        
        for i = 1:generic_negative_data
            filename = nega_info(i).name;
            out = load(filename);
            Xnega(:,i) = out.pyramid;
        end
    elseif isequal(features, 'mac') | isequal(features, 'rmac')
        nega_info = dir(fullfile(db_nega_path, '*.mc'));
        
        for i = 1:generic_negative_data
            filename = nega_info(i).name;
            out = read_mc(filename);
            Xnega(:,i) = out{1}';
        end
    elseif isequal(features(1:7), 'netvlad')
        nega_info = dir(fullfile(db_nega_path, '*.mc'));
        
        for i = 1:generic_negative_data
            filename = nega_info(i).name;
            out = read_mc(filename);
            Xnega(:,i) = out{1}';
        end
    end
    
    
    
else
    
for i = 1:generic_negative_data
    db_nega_path = [pre_nega_path Flickr_folders{i}];
    addpath(db_nega_path)
    content_nega = ls(db_nega_path);
    disp(i)
%     find_mc = strfind(content_nega, '.mc');
%     space1 = content_nega(end);
%     space2 = content_nega(find_mc(1)+3);
%     space3 = content_nega(find_mc(2)+3);
    space_idx = [0 find(isspace(content_nega))];
    %file_idx = sort([0 strfind(content_nega, space1) strfind(content_nega, space2) strfind(content_nega, space3)]);
    N = length(space_idx)-1;
    for j = 1:N
        filename = content_nega(space_idx(j)+1:space_idx(j+1)-1);
        if length(filename) > 1
            count = count+1;
            %disp(count)
            %disp(filename)
            out = read_mc(filename);
%              if count == 13253
%                  disp('stop')
%              end
            if ~isempty(out)
                Xnega(:,count) = out{1}';
            end
        end
    end
    if (count ~= sum(Flickr_sizes(1:i)))
        disp(['oops! something went wrong with Flickr folder number ' num2str(i)])
    end
end

end

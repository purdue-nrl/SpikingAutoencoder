% code to create paired audio dataset:
%% clear all

clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));
addpath(genpath('../DeepLearnToolbox'));
addpath(genpath('../utils'));
addpath(genpath('../SPEECH'));

%% Load data
% load mnist_uint8;
% train_x = double(train_x') / 255;
% test_x  = double(test_x')  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);
% 
% mnist_x{1} = train_x(:,find(train_y(:,1)==1));
% mnist_x{2} = train_x(:,find(train_y(:,2)==1));
% mnist_x{3} = train_x(:,find(train_y(:,3)==1));
% mnist_x{4} = train_x(:,find(train_y(:,4)==1));
% mnist_x{5} = train_x(:,find(train_y(:,5)==1));
% mnist_x{6} = train_x(:,find(train_y(:,6)==1));
% mnist_x{7} = train_x(:,find(train_y(:,7)==1));
% mnist_x{8} = train_x(:,find(train_y(:,8)==1));
% mnist_x{9} = train_x(:,find(train_y(:,9)==1));
% mnist_x{10} = train_x(:,find(train_y(:,10)==1));
% 
% save('mnist_x.mat', 'mnist_x');

%load mnist_x.mat;

K = ones(10,1);

imdb.image = [];
imdb.audio = {};
imdb.fs = [];
imdb.coch = [];

imdb.label = [];
imdb.set = [];
imdb.gender = [];

% Directory configuration
root_dir     = '/home/min/a/roy77/SNN/SPEECH/ti46_wav/ti10_digits';
root_subdir  = 'train';  % Change to 'test' to process audio files in the test directory
root_dir_use = sprintf('%s/%s', root_dir, root_subdir);

% Speech processing parameters
num_channels_data = 39;    % Number of channels in processed audio data
num_tsteps_data   = 1500;  % Number of time steps in processed audio data
dec_ratio         = 10;    % Decimation ratio to downsample the original audio data
earQ              = 8;
stepfactor        = 16/32;

% Assign a seed for the random number generators in the code
rand('seed', 0);

%% FEMALE TRAIN SET
% List female speech recordings
num_spkrs    = 8;
file_count_f = 0;
file_list_f = {};

for spkr = 1:num_spkrs
    speech_dir        = sprintf('%s/f%d/*.wav', root_dir_use, spkr);
    file_list_f{spkr} = dir(speech_dir);
    file_count_f      = file_count_f + numel(file_list_f{spkr});
end


% Process speech data using the Lyon Passive Ear model of cochlea
imdb.coch    = zeros(num_channels_data, num_tsteps_data, file_count_f);
imdb.label  = zeros(file_count_f, 1);
file_indx = 1;



for spkr = 1:num_spkrs
    fprintf('Female train set speaker %1.0f/%1.0f \n', spkr, num_spkrs);
    for file_num = 1:numel(file_list_f{spkr})
        %fprintf('%s\n', file_list_f{spkr}(file_num).name);
        [speech, fs] = audioread(file_list_f{spkr}(file_num).name);
        coch        = LyonPassiveEar(speech, fs, dec_ratio, earQ, stepfactor);
        coch        = coch / max(max(coch));
        
        
        % Assert if the number of channels in cochlea matches with that
        % specified in the speech processing configuration parameter
        num_channels_coch = size(coch, 1);
        assert(num_channels_coch == num_channels_data);
        
        % Save the Lyon Passive Ear coefficients in the processed audio-data array
        num_tsteps_coch = size(coch, 2);
        
        if(num_tsteps_coch <= num_tsteps_data)
           imdb.coch(:, 1:num_tsteps_coch, file_indx) = coch;
        else
           imdb.coch(:, :, file_indx) = coch(:, 1:num_tsteps_data);
        end
        
        % Assign the output label corresponding to the processed audio-data
        
        imdb.label(file_indx) = str2double(file_list_f{spkr}(file_num).name(2));
        
        % Increment the file index
              
        imdb.audio{end+1} = fullfile(file_list_f{spkr}(file_num).folder, file_list_f{spkr}(file_num).name);
        imdb.fs(end+1) = fs;
        imdb.image(:,end+1) = mnist_x{imdb.label(file_indx)+1}(:,K(imdb.label(file_indx)+1));
        imdb.set(end+1) = 0;     
        imdb.gender(end+1) = 0;
        K(imdb.label(file_indx)+1) = K(imdb.label(file_indx)+1) + 1;
        
        file_indx = file_indx + 1;
        
        
    end
end

%% FEMALE TEST SET

% Directory configuration
root_subdir  = 'test';  % Change to 'test' to process audio files in the test directory
root_dir_use = sprintf('%s/%s', root_dir, root_subdir);

% Assign a seed for the random number generators in the code
rand('seed', 0);

% List female speech recordings
num_spkrs    = 8;
file_count_f = 0;
file_list_f = {};

for spkr = 1:num_spkrs
    speech_dir        = sprintf('%s/f%d/*.wav', root_dir_use, spkr);
    file_list_f{spkr} = dir(speech_dir);
    file_count_f      = file_count_f + numel(file_list_f{spkr});
end


% Process speech data using the Lyon Passive Ear model of cochlea
imdb.coch    = cat(3,imdb.coch, zeros(num_channels_data, num_tsteps_data, file_count_f));
imdb.label  = cat(1,imdb.label, zeros(file_count_f, 1));

for spkr = 1:num_spkrs
    fprintf('Female test set speaker  %1.0f/%1.0f \n', spkr, num_spkrs);
    for file_num = 1:numel(file_list_f{spkr})
        
        [speech, fs] = audioread(file_list_f{spkr}(file_num).name);
        coch        = LyonPassiveEar(speech, fs, dec_ratio, earQ, stepfactor);
        coch        = coch / max(max(coch));
        
        % Assert if the number of channels in cochlea matches with that
        % specified in the speech processing configuration parameter
        num_channels_coch = size(coch, 1);
        assert(num_channels_coch == num_channels_data);
        
        % Save the Lyon Passive Ear coefficients in the processed audio-data array
        num_tsteps_coch = size(coch, 2);
        
        if(num_tsteps_coch <= num_tsteps_data)
           imdb.coch(:, 1:num_tsteps_coch, file_indx) = coch;
        else
           imdb.coch(:, :, file_indx) = coch(:, 1:num_tsteps_data);
        end
        
        % Assign the output label corresponding to the processed audio-data
        imdb.label(file_indx) = str2double(file_list_f{spkr}(file_num).name(2));
        
        % Increment the file index
        
        
        imdb.audio{end+1} = fullfile(file_list_f{spkr}(file_num).folder, file_list_f{spkr}(file_num).name);
        imdb.fs(end+1) = fs;   
        imdb.image(:,end+1) = mnist_x{imdb.label(file_indx)+1}(:,K(imdb.label(file_indx)+1));
        imdb.set(end+1) = 1;     
        imdb.gender(end+1) = 0;
        K(imdb.label(file_indx)+1) = K(imdb.label(file_indx)+1) + 1;
        
        file_indx = file_indx + 1;
    end
end

%% MALE train set 

% Directory configuration
root_subdir  = 'train';  % Change to 'test' to process audio files in the test directory
root_dir_use = sprintf('%s/%s', root_dir, root_subdir);

% Assign a seed for the random number generators in the code
rand('seed', 0);

% List male speech recordings
num_spkrs    = 8;
file_count_m = 0;
file_list_m = {};

for spkr = 1:num_spkrs
    speech_dir        = sprintf('%s/m%d/*.wav', root_dir_use, spkr);
    file_list_m{spkr} = dir(speech_dir);
    file_count_m      = file_count_m + numel(file_list_m{spkr});
end


% Process speech data using the Lyon Passive Ear model of cochlea
imdb.coch    = cat(3,imdb.coch, zeros(num_channels_data, num_tsteps_data, file_count_m));
imdb.label  = cat(1,imdb.label, zeros(file_count_m, 1));

for spkr = 1:num_spkrs
    fprintf('Male train set speaker %1.0f/%1.0f \n', spkr, num_spkrs);
    for file_num = 1:numel(file_list_m{spkr})
        [speech, fs] = audioread(file_list_m{spkr}(file_num).name);
        coch        = LyonPassiveEar(speech, fs, dec_ratio, earQ, stepfactor);
        coch        = coch / max(max(coch));
        
        % Assert if the number of channels in cochlea matches with that
        % specified in the speech processing configuration parameter
        num_channels_coch = size(coch, 1);
        assert(num_channels_coch == num_channels_data);
        
        % Save the Lyon Passive Ear coefficients in the processed audio-data array
        num_tsteps_coch = size(coch, 2);
        
        if(num_tsteps_coch <= num_tsteps_data)
           imdb.coch(:, 1:num_tsteps_coch, file_indx) = coch;
        else
           imdb.coch(:, :, file_indx) = coch(:, 1:num_tsteps_data);
        end
        
        % Assign the output label corresponding to the processed audio-data
        imdb.label(file_indx) = str2double(file_list_m{spkr}(file_num).name(2));
        
        % Increment the file index
        
      
        imdb.audio{end+1} = fullfile(file_list_m{spkr}(file_num).folder, file_list_m{spkr}(file_num).name);
        imdb.fs(end+1) = fs;
        imdb.image(:,end+1) = mnist_x{imdb.label(file_indx)+1}(:,K(imdb.label(file_indx)+1));
        imdb.set(end+1) = 0;     
        imdb.gender(end+1) = 1;
        K(imdb.label(file_indx)+1) = K(imdb.label(file_indx)+1) + 1;
        
        file_indx = file_indx + 1;
    end
end

%% MALE test set 

% Directory configuration
root_subdir  = 'test';  % Change to 'test' to process audio files in the test directory
root_dir_use = sprintf('%s/%s', root_dir, root_subdir);

% Assign a seed for the random number generators in the code
rand('seed', 0);

% List male speech recordings
num_spkrs    = 8;
file_count_m = 0;
file_list_m = {};

for spkr = 1:num_spkrs
    speech_dir        = sprintf('%s/m%d/*.wav', root_dir_use, spkr);
    file_list_m{spkr} = dir(speech_dir);
    file_count_m      = file_count_m + numel(file_list_m{spkr});
end


% Process speech data using the Lyon Passive Ear model of cochlea
imdb.coch    = cat(3, imdb.coch, zeros(num_channels_data, num_tsteps_data, file_count_m));
imdb.label  = cat(1, imdb.label, zeros(file_count_m, 1));



for spkr = 1:num_spkrs
    fprintf('Male test set speaker %1.0f/%1.0f\n', spkr, num_spkrs);
    for file_num = 1:numel(file_list_m{spkr})
        [speech, fs] = audioread(file_list_m{spkr}(file_num).name);
        coch        = LyonPassiveEar(speech, fs, dec_ratio, earQ, stepfactor);
        coch        = coch / max(max(coch));
        
        % Assert if the number of channels in cochlea matches with that
        % specified in the speech processing configuration parameter
        num_channels_coch = size(coch, 1);
        assert(num_channels_coch == num_channels_data);
        
        % Save the Lyon Passive Ear coefficients in the processed audio-data array
        num_tsteps_coch = size(coch, 2);
        
        if(num_tsteps_coch <= num_tsteps_data)
           imdb.coch(:, 1:num_tsteps_coch, file_indx) = coch;
        else
           imdb.coch(:, :, file_indx) = coch(:, 1:num_tsteps_data);
        end
        
        % Assign the output label corresponding to the processed audio-data
        imdb.label(file_indx) = str2double(file_list_m{spkr}(file_num).name(2));
        
        % Increment the file index
        
        imdb.audio{end+1} = fullfile(file_list_m{spkr}(file_num).folder, file_list_m{spkr}(file_num).name);
        imdb.fs(end+1) = fs;
        imdb.image(:,end+1) = mnist_x{imdb.label(file_indx)+1}(:,K(imdb.label(file_indx)+1));
        imdb.set(end+1) = 1;     
        imdb.gender(end+1) = 1;
        K(imdb.label(file_indx)+1) = K(imdb.label(file_indx)+1) + 1;
        
        file_indx = file_indx + 1;
    end
end


imdb.coch = reshape(imdb.coch, size(imdb.coch,1)*size(imdb.coch,2), size(imdb.coch,3));
save('./dataset/mnist_multi_modal.mat', 'imdb', '-v7.3');


N = numel(imdb.fs);
% shuffle all data
idx = randperm(N);
imdb.audio = imdb.audio(idx);
imdb.gender = imdb.gender(1,idx);
imdb.coch = imdb.coch(:,idx);
imdb.image = imdb.image(:,idx);
imdb.label = imdb.label(idx);
imdb.set = [zeros(1, 3500), ones(1, N-3500)];

save('./dataset/mnist_multimodal_84_16.mat', 'imdb', '-v7.3');
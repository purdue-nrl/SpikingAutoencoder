%% clear all

clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));

%% define parameters(opts)
opts.dt                 = 0.001;
opts.tau                = 0.01;
opts.max_rate           = 20;
opts.duration           = 0.030;
opts.batch_size         = 50;
opts.threshold          = 1;
opts.t_ref              = 0;
opts.neuron_model       = 'LIF';
opts.rounds             = 1;
opts.alpha              = 5e-5;
opts.scale              = 1;
opts.grad_clip          = false;
opts.grad_clip_thresh   = 100;
opts.adam               = true;
opts.beta1              = 0.9;
opts.beta2              = 0.999;
opts.epsilon            = 10e-8;
opts.numepochs          = 20;
opts.weight_decay       = 1e-4;
opts.continue           = 1;
opts.train              = [];
opts.val                = [];
opts.T                  = 15;
opts.aeDir = './output_final/AC_MNIST/';
opts.expDir             ='/home/nano01/a/roy77/spiking_autoencoder/output_final/AC_MNIST/one-image-per-class/hidden_state/15/';
opts.evaluate           = false;

if ~exist(opts.expDir, 'dir') 
    mkdir(opts.expDir);
end
save(fullfile(opts.expDir, 'opts.mat'), 'opts');

%% Load dataset

load(fullfile(opts.aeDir,'ae.mat'));

if ~exist(fullfile(opts.aeDir, 'one-image-per-class','imdb_15T_v2.mat'), 'file')
    fprintf('generating dataset \n')
    load mnist_multimodal_84_16_v2.mat
    ae_opts = opts;
    ae_opts.duration = 0.015;
    ae_opts.max_rate = 300;
    imdb = generate_hidden_state(imdb, ae, ae_opts);
    save(fullfile(opts.aeDir, 'one-image-per-audio','imdb_15T_v2.mat'), 'imdb');    
else 
    fprintf('loading dataset \n')
    load(fullfile(opts.aeDir, 'one-image-per-class', 'imdb_15T_v2.mat'));    
end

% if ~exist(fullfile(opts.aeDir, 'one-image-per-audio','imdb_15T.mat'), 'file')
%     fprintf('generating dataset \n')
%     load mnist_multimodal_84_16.mat
%     ae_opts = opts;
%     ae_opts.duration = 0.015;
%     ae_opts.max_rate = 300;
%     imdb = generate_hidden_state(imdb, ae, ae_opts);
%     save(fullfile(opts.aeDir, 'one-image-per-audio','imdb_15T.mat'), 'imdb');    
% else 
%     fprintf('loading dataset \n')
%     load(fullfile(opts.aeDir, 'one-image-per-audio', 'imdb_15T.mat'));    
% end
ae = ae.initialize(opts);


% setup audio-coder
ac = ac_setup([39*1500, 2048, 196]);

[ac, stats] = ac_train_v2(ac, ae, imdb, opts);

% for one-image-per-audio use ac_train(ac,ae,imdb,opts)


    
       



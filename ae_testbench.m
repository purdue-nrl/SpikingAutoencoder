%% clear all

clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));
addpath(genpath('../DeepLearnToolbox'));
addpath(genpath('../utils'));

%% Load data

% dataset MNIST
load mnist_uint8;
train_x = double(train_x') / 255;
test_x  = double(test_x')  / 255;
train_y = double(train_y);
test_y  = double(test_y);

% dataset F-MNIST
% load f-mnist;

%% Set global variables
opts.dt                 = 0.001;
opts.tau                = 0.01;
opts.max_rate           = 300;
opts.duration           = 0.005;
opts.batch_size         = 100;
opts.threshold          = 1;
opts.t_ref              = 2*opts.dt;
opts.neuron_model       = 'LIF';
opts.rounds             = 1;
opts.alpha              = 5e-4;
opts.scale              = 1;
opts.grad_clip          = false;
opts.grad_clip_thresh   = 100;
opts.adam               = true;
opts.beta1              = 0.9;
opts.beta2              = 0.999;
opts.epsilon            = 10e-8;
opts.numepochs          = 1;
opts.weight_decay       = 1e-4;
opts.continue           = 1;
opts.mask               = 'bitxor';
%% initialize the two auto-encoders

ae = auto_encoder(784, 196, 784);
% figure(2);
% subplot(2,2,1);
% imagesc(ae.weights_code);
% colormap(gray);
% colorbar;
% title('Initialized Weight-Code');
% subplot(2,2,2);
% imagesc(ae.weights_decode);
% colormap(gray);
% colorbar;
% title('Initialized Weight-Decode');

output_dir = './output_final/AE_MNIST/timesteps/5/';
if ~exist(output_dir, 'dir'), mkdir(output_dir) ; end

%% train ae 
train_examples      = size(train_x,2);
test_examples       = size(test_x, 2);
train_loss          = [];
test_loss           = [];
train_mse           = [];
test_mse            = [];

start = opts.continue * findLastCheckpoint(output_dir) ;

modelPath = @(ep) fullfile(output_dir, sprintf('ae-epoch-%d.mat', ep));
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start);
  [ae, train_loss, test_loss, train_mse, test_mse] = loadState(modelPath(start));
end

f = figure('Name', output_dir);


for i = start+1:opts.numepochs
    train_x = train_x(:,randperm(train_examples));
    ae.batch_loss = [];
    num = 0;
    acc_loss = 0;
    acc_mse = 0;
    for t = 1:opts.batch_size:train_examples
        fprintf('train : epoch %02d: %3d/%3d:', i, ... 
        fix((t-1)/opts.batch_size)+1, ceil(train_examples/opts.batch_size)) ;
        batchSize = min(opts.batch_size, train_examples - t + 1);  
        batchStart = t;
        batchEnd = min(t+opts.batch_size-1, train_examples);
        [ae, loss, ae_mse] = ae_train(ae, train_x(:,batchStart:batchEnd), opts);
        num = num + batchSize;
        acc_loss = acc_loss + loss;
        acc_mse = acc_mse + ae_mse;
        ae.batch_loss(fix((t-1)/opts.batch_size)+1) = loss/batchSize;  
        ae.avg_loss(fix((t-1)/opts.batch_size)+1) = acc_loss/num;
        ae.batch_mse(fix((t-1)/opts.batch_size)+1) = ae_mse;
        ae.avg_mse(fix((t-1)/opts.batch_size)+1) = acc_mse/(fix((t-1)/opts.batch_size)+1);
        fprintf(' loss: %1.4f avg_loss: %1.4f mse: %1.4f \n', ...
                  ae.batch_loss(fix((t-1)/opts.batch_size)+1), ...
                  ae.avg_loss(fix((t-1)/opts.batch_size)+1), ...
                  ae.batch_mse(fix((t-1)/opts.batch_size)+1));
    end
    
     
    %total reconstruction loss on training set 
    opts2 = opts;
    opts2.batch_size = train_examples;
    ae = ae.initialize(opts2);
    [ae, loss, ae_mse] = ae_test(ae, train_x, opts2);
    train_loss(i) = loss/train_examples;
    train_mse(i) = ae_mse;
    fprintf('Train_loss = %1.4f Train_mse = %1.4f', train_loss(i), train_mse(i));
    %total reconstruction loss on testing set 
    opts2 = opts;
    opts2.batch_size = test_examples;
    ae = ae.initialize(opts2);
    [ae, loss, ae_mse] = ae_test(ae, test_x, opts2);
    test_loss(i) = loss/test_examples;
    test_mse(i) = ae_mse;
    fprintf('  Test_loss = %1.4f Test_mse = %1.4f\n', test_loss(i), test_mse(i));
    
    % pick a random image from training set
    idx1 = randi([1, train_examples],1);   
  
    % convert that image into spike train
    spike_input = pixel_to_spike(train_x(:,idx1), opts.dt, opts.duration, opts.max_rate);
    %feed it to the trained network
    opts2 = opts;
    opts2.batch_size = 1;
    ae = ae.initialize(opts2);
    train_output_spikes = zeros(size(spike_input,1));
    for n = 1:opts.duration/opts.dt
        ae = ae.code(spike_input(:,:,n), opts);
        ae = ae.decode(opts);
        %output_spikes = bitor(output_spikes,ae.output.spikes);
        train_output_spikes = train_output_spikes + ae.output.spikes;
    end

    % pick a random image from testing set
    idx2 = randi([1,test_examples],1);
    spike_input = pixel_to_spike(test_x(:,idx2), opts.dt, opts.duration, opts.max_rate);
    opts2 = opts;
    opts2.batch_size = 1;
    ae = ae.initialize(opts2);
    test_output_spikes = zeros(size(spike_input,1));

    for n = 1:opts.duration/opts.dt
        ae = ae.code(spike_input(:,:,n), opts);
        ae = ae.decode(opts);
        %output_spikes = bitor(output_spikes,ae.output.spikes);
        test_output_spikes = test_output_spikes + ae.output.spikes;
    end

    x = 1:1:i;
    ae_fig_plot( f, train_x(:,idx1), train_output_spikes(:,1), ...
                  test_x(:,idx2), test_output_spikes(:,1), ...
                  ae.weights_code, ae.weights_decode, ...
                  train_loss, test_loss, ...
                  train_mse, test_mse, x', output_dir);  
    save(fullfile(output_dir,sprintf('ae-epoch-%d.mat', i)),'ae', 'train_loss', 'test_loss', 'train_mse', 'test_mse');
    
end
        














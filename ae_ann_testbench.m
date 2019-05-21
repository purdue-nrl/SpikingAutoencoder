%% clear all
clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));
addpath(genpath('../DeepLearnToolbox'));
addpath(genpath('../utils'));

%% Load data
% Load MNIST
% load mnist_uint8;
% train_x = double(train_x')/255;
% test_x  = double(test_x')/255;
% train_y = double(train_y);
% test_y  = double(test_y);

% Load fashion-mnist
load f-mnist;

%% Set global variables
opts.batch_size         = 100;
opts.alpha              = 0.005;
opts.grad_clip          = false;
opts.grad_clip_thresh   = 100;
opts.adam               = true;
opts.beta1              = 0.9;
opts.beta2              = 0.999;
opts.epsilon            = 10e-8;
opts.numepochs          = 5;
opts.weight_decay       = 1e-4;
opts.continue           = 0;
%% initialize the autoencoder

ae = auto_encoder_ann(784, 1024, 784);
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

output_dir = './output_final/AE_FMNIST/test/lr/0.005';
if ~exist(output_dir, 'dir'), mkdir(output_dir) ; end

%% train ae 
train_examples      = size(train_x,2);
N                   = train_examples;
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

ae.initialize(opts);
for i = start+1:opts.numepochs
    train_x = train_x(:,randperm(train_examples));
    ae.batch_loss = [];
    num = 0;
    acc_loss = 0;
    acc_mse = 0;
    for t = 1:opts.batch_size:N
        fprintf('train : epoch %02d: %3d/%3d:', i, ... 
        fix((t-1)/opts.batch_size)+1, ceil(N/opts.batch_size)) ;
        batchSize = min(opts.batch_size, N - t + 1);  
        batchStart = t;
        batchEnd = min(t+opts.batch_size-1, N) ;
        opts.batch_number = fix((t-1)/opts.batch_size)+1;
        [ae, loss, ae_mse] = ae_ann_train(ae, train_x(:,batchStart:batchEnd), opts);
        num = num + batchSize;
        
        acc_loss = acc_loss + loss;
        acc_mse = acc_mse + ae_mse;
        ae.batch_loss(opts.batch_number) = loss/batchSize;  
        ae.avg_loss(opts.batch_number) = acc_loss/num;
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
    [ae, loss, ae_mse] = ae_ann_test(ae, train_x, opts2);
    train_loss(i) = loss/train_examples;
    train_mse(i) = ae_mse;
    fprintf('Train_loss = %1.4f ', train_loss(i))
    
    %total reconstruction loss on testing set 
    opts2 = opts;
    opts2.batch_size = test_examples;
    ae = ae.initialize(opts2);
    [ae, loss, ae_mse] = ae_ann_test(ae, test_x, opts2);
    test_loss(i) = loss/test_examples;
    test_mse(i) = ae_mse;
    fprintf('Test_loss = %1.4f \n', test_loss(i))
    fprintf('Test_mse = %1.4f \n', test_mse(i))
    
    % pick a random image from training set
    idx1 = randi([1, train_examples],1);   
    opts2 = opts;
    opts2.batch_size = 1;
    ae = ae.initialize(opts2);
    ae = ae.code(train_x(:,idx1));
    ae = ae.decode();
    train_output = ae.output.a;
   

    % pick a random image from testing set
    idx2 = randi([1,test_examples],1);
    opts2 = opts;
    opts2.batch_size = 1;
    ae = ae.initialize(opts2);
    
    ae = ae.code(test_x(:,idx2));
    ae = ae.decode();
    test_output = ae.output.a;
    
    x = 1:1:i;
    ae_fig_plot( f, train_x(:,idx1), train_output(:,1), ...
                  test_x(:,idx2), test_output(:,1), ...
                  ae.weights_code, ae.weights_decode, ...
                  train_loss, test_loss, ...
                  train_mse, test_mse, x', output_dir);  
    save(fullfile(output_dir,sprintf('ae-epoch-%d.mat', i)),'ae', 'train_loss', 'test_loss', 'train_mse', 'test_mse');
    
end
        














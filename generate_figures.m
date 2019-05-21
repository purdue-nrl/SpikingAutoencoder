% Date: 03/26/2018
% Author: Deboleena Roy

%% clear all

clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));
addpath(genpath('../DeepLearnToolbox'));
addpath(genpath('../utils'));

%% Load data
load mnist_uint8;
train_x = double(train_x') / 255;
test_x  = double(test_x')  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% Set global variables
opts.dt = 0.001;
opts.tau = 1;
opts.max_rate = 300;
opts.duration = 0.010;
opts.batch_size = 100;
opts.threshold = 1;
opts.t_ref = 2*opts.dt;
opts.neuron_model = 'IF';
opts.rounds = 1;
opts.alpha = 0.005;
opts.scale = 1;
opts.grad_clip = true;
opts.grad_clip_thresh = 100;
opts.adam = true;
opts.beta1 = 0.9;
opts.beta2 = 0.999;
opts.epsilon = 10e-8;
opts.numepochs = 5;
opts.weight_decay = 1e-4;
opts.continue = 1;

load './output/LIF-196-bitxor-5e-3/ae-epoch-1.mat';


% 
train_examples = size(train_x,2);
% N = train_examples;
test_examples = size(test_x, 2);


    
%total reconstruction loss on training set
% opts2 = opts;
% opts2.batch_size = train_examples;
% ae = ae.initialize(opts2);
% [ae, loss] = ae_test(ae, train_x, opts2);
% trainLoss = loss/train_examples;
% fprintf('Train_loss = %1.4f ', trainLoss)

%total reconstruction loss on testing set
% opts2 = opts;
% opts2.batch_size = test_examples;
% ae = ae.initialize(opts2);
% [ae, loss] = ae_test(ae, test_x, opts2);
% testLoss = loss/test_examples;
% fprintf('Test_loss = %1.4f \n', testLoss)


% pick a random image from training set
idx = randi([1, train_examples],1);
% display it
switchFigure(1);
subplot(2,2,1);
imagesc(reshape(train_x(:,idx), 28,28)'); colormap(gray); colorbar;
% convert that image into spike train
spike_input = pixel_to_spike(train_x(:,idx), opts.dt, opts.duration, opts.max_rate);
%feed it to the trained network
opts2 = opts;
opts2.batch_size = 1;
ae = ae.initialize(opts2);
output_spikes = zeros(size(spike_input,1));
for n = 1:opts.duration/opts.dt
    ae = ae.code(spike_input(:,:,n), opts);
    ae = ae.decode(opts);
    output_spikes = output_spikes + ae.output.spikes;    
end
output_spikes = output_spikes/max(output_spikes);
% display reconstructed image
switchFigure(1);
subplot(2,2,2);
imagesc(reshape(output_spikes(:,1), 28, 28)'); colorbar; colormap(gray);

% pick a random image from testing set
idx = randi([1,test_examples],1);
% display it
switchFigure(1);
subplot(2,2,3);
imagesc(reshape(test_x(:,idx), 28,28)'); colormap(gray); colorbar;
spike_input = pixel_to_spike(test_x(:,idx), opts.dt, opts.duration, opts.max_rate);
opts2 = opts;
opts2.batch_size = 1;
ae = ae.initialize(opts2);
output_spikes = zeros(size(spike_input,1));

for n = 1:opts.duration/opts.dt
    ae = ae.code(spike_input(:,:,n), opts);
    ae = ae.decode(opts);
    output_spikes = output_spikes + ae.output.spikes;
    
end
output_spikes = output_spikes/max(output_spikes);
switchFigure(1);
subplot(2,2,4);
imagesc(reshape(output_spikes(:,1), 28, 28)'); colorbar; colormap(gray);




switchFigure(2);
subplot(2,1,1);
imagesc(ae.weights_code);
colormap(gray);
colorbar;
title('Trained Weight-Code');
subplot(2,1,2);
imagesc(ae.weights_decode);
colormap(gray);
colorbar;
title('Trained Weight-Decode');



















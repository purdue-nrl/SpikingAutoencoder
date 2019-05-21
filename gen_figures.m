% Use this to generate figures for the paper
%clear all;
close all;
addpath(genpath('./dataset/'));
addpath(genpath('../SPEECH'));



%% Figure 1 : Input image to Spike input transformation
load mnist_uint8;
train_x = double(train_x') / 255;
idx1 = 1;
opts.dt                 = 0.001;
opts.max_rate           = 300;
opts.duration           = 0.015;
train_x_spike = pixel_to_spike(train_x(:,idx1), opts.dt, opts.duration, opts.max_rate);
% figure(1)
% imagesc(reshape(train_x(:,idx1), 28, 28)'); colormap(gray); pbaspect([1 1 1]);
% figure(2)
% imagesc(reshape(train_x_spike(:,:,1), 28, 28)'); colormap(gray); pbaspect([1 1 1]);
% figure(3)
% imagesc(reshape(train_x_spike(:,:,2), 28, 28)'); colormap(gray); pbaspect([1 1 1]);
% figure(4)
% imagesc(reshape(train_x_spike(:,:,3), 28, 28)'); colormap(gray); pbaspect([1 1 1]);
% figure(5)
% imagesc(reshape(sum(train_x_spike(:,:,:),3), 28, 28)'); colormap(gray); pbaspect([1 1 1]);

%% Figure 2 : Audio Input transformation
%load mnist_multimodal_84_16_v2.mat

% idx1 = 1;
% num_channels_data = 39;    % Number of channels in processed audio data
% num_tsteps_data   = 1500;  % Number of time steps in processed audio data
% dec_ratio         = 10;    % Decimation ratio to downsample the original audio data
% earQ              = 8;
% stepfactor        = 16/32;
% 
% [audio, fs] = audioread(cell2mat(imdb.audio(:,idx1))) ;
% coch        = LyonPassiveEar(audio, fs, dec_ratio, earQ, stepfactor);
% %plot(audio);
% imagesc(coch);

load './output_final/AC_MNIST/ae.mat'
opts.batch_size = 1;
opts.neuron_model = 'LIF';
opts.tau = 0.01;
opts.threshold = 1;
ae = ae.initialize(opts);
train_output_spikes = zeros(784,1);
train_hidden_spikes = zeros(196,1);
for n = 1:opts.duration/opts.dt
    ae = ae.code(train_x_spike(:,:,n), opts);
    train_hidden_spikes = train_hidden_spikes + ae.hidden.spikes;
    ae = ae.decode(opts);
    %output_spikes = bitor(output_spikes,ae.output.spikes);
    train_output_spikes = train_output_spikes + ae.output.spikes;
end

figure(1)
imagesc(reshape(train_output_spikes, 28, 28)'); colormap(gray); pbaspect([1 1 1]);
figure(2)
imagesc(reshape(train_hidden_spikes, 14, 14)'); colormap(gray); pbaspect([1 1 1]);






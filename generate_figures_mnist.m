clear all;
close all;
%% Load path
addpath(genpath('./dataset/'));
load('./dataset/mnist_uint8.mat');

%%Load dataset
train_x = double((train_x)'/255);
test_x  = double((test_x)'/255);
train_y = double(train_y);
test_y  = double(test_y);

mnist_train_x{ 1} = train_x(:,find(train_y(:, 1)==1));
mnist_train_x{ 2} = train_x(:,find(train_y(:, 2)==1));
mnist_train_x{ 3} = train_x(:,find(train_y(:, 3)==1));
mnist_train_x{ 4} = train_x(:,find(train_y(:, 4)==1));
mnist_train_x{ 5} = train_x(:,find(train_y(:, 5)==1));
mnist_train_x{ 6} = train_x(:,find(train_y(:, 6)==1));
mnist_train_x{ 7} = train_x(:,find(train_y(:, 7)==1));
mnist_train_x{ 8} = train_x(:,find(train_y(:, 8)==1));
mnist_train_x{ 9} = train_x(:,find(train_y(:, 9)==1));
mnist_train_x{10} = train_x(:,find(train_y(:,10)==1));

mnist_test_x{ 1} = test_x(:,find(test_y(:, 1)==1));
mnist_test_x{ 2} = test_x(:,find(test_y(:, 2)==1));
mnist_test_x{ 3} = test_x(:,find(test_y(:, 3)==1));
mnist_test_x{ 4} = test_x(:,find(test_y(:, 4)==1));
mnist_test_x{ 5} = test_x(:,find(test_y(:, 5)==1));
mnist_test_x{ 6} = test_x(:,find(test_y(:, 6)==1));
mnist_test_x{ 7} = test_x(:,find(test_y(:, 7)==1));
mnist_test_x{ 8} = test_x(:,find(test_y(:, 8)==1));
mnist_test_x{ 9} = test_x(:,find(test_y(:, 9)==1));
mnist_test_x{10} = test_x(:,find(test_y(:,10)==1));

load('./output_final/AE_MNIST/mask/bitxor/ae-epoch-1.mat');
opts.dt                 = 0.001;
opts.tau                = 0.01;
opts.max_rate           = 300;
opts.duration           = 0.015;
opts.batch_size         = 1;
opts.threshold          = 1;
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
opts.numepochs          = 2;
opts.weight_decay       = 1e-4;
opts.continue           = 1;
opts.mask               = 'bitxor';
opts.save               = './figures/mnist/';

if ~exist(opts.save, 'dir'), mkdir(opts.save) ; end

for i = 1:10
    for j = 1:10
        Input(i,j,:,:) = mat2gray(reshape((mnist_test_x{i}(:,j)), 28, 28)'); 
        ae.initialize(opts);
        spike_input = pixel_to_spike(mnist_test_x{i}(:,j), opts.dt, opts.duration, opts.max_rate);
        output_spikes = zeros(784,1);
        for t = 1:opts.duration/opts.dt
            ae = ae.code_test(spike_input(:,:,t), opts);
            ae = ae.decode_test(opts);
            output_spikes = output_spikes + ae.output.spikes;            
        end
        output_spikes = output_spikes/max(output_spikes);
        figure(1);
        subplot(1,3,1);
        imagesc(reshape(mnist_test_x{i}(:,j), 28, 28)'); colormap('gray');drawnow;
        subplot(1,3,2);
        imagesc(reshape(sum(spike_input,3), 28, 28)'); colormap('gray'); drawnow;
        subplot(1,3,3);
        imagesc(reshape(output_spikes, 28, 28)'); colormap('gray'); drawnow;
        filename=fullfile(opts.save, 'test_output', sprintf('test_output-%d-%d.tif', i, j));
        imwrite(mat2gray(reshape(output_spikes, 28, 28)'), filename);     
        filename=fullfile(opts.save, 'test_input', sprintf('test_input-%d-%d.tif', i, j));
        imwrite(squeeze(Input(i,j,:,:)), filename); 
        filename=fullfile(opts.save, 'test_input_spike', sprintf('test_input_spike-%d-%d.tif', i, j));
        imwrite(mat2gray(reshape(sum(spike_input,3),28,28)'), filename);
    end
end








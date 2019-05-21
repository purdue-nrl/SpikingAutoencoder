clear all;
close all;
%% Load path
addpath(genpath('./dataset/'));
load('./dataset/mnist_multimodal_84_16.mat');

%%Load dataset
train_x = imdb.coch(:,imdb.set(1,:)==0);
test_x  = imdb.coch(:,imdb.set(1,:)==1);
train_y = imdb.label(imdb.set(1,:)==0, 1);
test_y  = imdb.label(imdb.set(1,:)==1, 1);

mnist_train_x{ 1} = train_x(:,find(train_y(:)==0));
mnist_train_x{ 2} = train_x(:,find(train_y(:)==1));
mnist_train_x{ 3} = train_x(:,find(train_y(:)==2));
mnist_train_x{ 4} = train_x(:,find(train_y(:)==3));
mnist_train_x{ 5} = train_x(:,find(train_y(:)==4));
mnist_train_x{ 6} = train_x(:,find(train_y(:)==5));
mnist_train_x{ 7} = train_x(:,find(train_y(:)==6));
mnist_train_x{ 8} = train_x(:,find(train_y(:)==7));
mnist_train_x{ 9} = train_x(:,find(train_y(:)==8));
mnist_train_x{10} = train_x(:,find(train_y(:)==9));

mnist_test_x{ 1} = test_x(:,find(test_y(:)==0));
mnist_test_x{ 2} = test_x(:,find(test_y(:)==1));
mnist_test_x{ 3} = test_x(:,find(test_y(:)==2));
mnist_test_x{ 4} = test_x(:,find(test_y(:)==3));
mnist_test_x{ 5} = test_x(:,find(test_y(:)==4));
mnist_test_x{ 6} = test_x(:,find(test_y(:)==5));
mnist_test_x{ 7} = test_x(:,find(test_y(:)==6));
mnist_test_x{ 8} = test_x(:,find(test_y(:)==7));
mnist_test_x{ 9} = test_x(:,find(test_y(:)==8));
mnist_test_x{10} = test_x(:,find(test_y(:)==9));

load('./output_final/AC_MNIST/ae.mat');
load('./output_final/AC_MNIST/one-image-per-audio/ac-epoch-20.mat');
opts.dt                 = 0.001;
opts.tau                = 0.01;
opts.max_rate           = 20;
opts.duration           = 0.030;
opts.batch_size         = 1;
opts.threshold          = 1;
opts.neuron_model       = 'LIF';
opts.rounds             = 1;
opts.scale              = 1;
opts.continue           = 1;
opts.save               = './figures/ac_mnist/one-image-per-audio/test_output';

if ~exist(opts.save, 'dir'), mkdir(opts.save) ; end

for i = 1:10
    for j = 1:5
        ac_initialize(ac, opts);
        ae.initialize(opts);
        spike_input = pixel_to_spike(mnist_test_x{i}(:,j), opts.dt, opts.duration, opts.max_rate);
        output_spikes = zeros(784,1);
        for t = 1:opts.duration/opts.dt
            ac = ac_ff_test(ac, spike_input(:,:,t), opts);
            ae.hidden.spikes = ac.layers{2}.spikes;
            ae = ae.decode_test(opts);
            output_spikes = output_spikes + ae.output.spikes; 
            %imagesc(reshape(output_spikes, 28, 28)'); colormap('gray'); drawnow;
        end
        figure(1);
        imagesc(reshape(output_spikes, 28, 28)'); colormap('gray'); drawnow;
        filename=fullfile(opts.save, sprintf('test_output-%d-%d.tif', i, j));
        imwrite(mat2gray(reshape(output_spikes, 28, 28)'), filename);     
    end
end








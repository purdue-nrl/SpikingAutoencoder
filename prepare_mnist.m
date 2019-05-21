% Date: 07/27/2018
% Author: Deboleena Roy

%% clear all

clear all;
close all;

%% Load path
addpath(genpath('./dataset/'));
addpath(genpath('../DeepLearnToolbox'));
addpath(genpath('../utils'));


% load mnist_uint8;
% test_x  = double(test_x')/ 255;
% test_y  = double(test_y);
% test_x = zscore(test_x);
%  
% mnist_x{1}  = test_x(:,test_y(:,1) ==1);
% mnist_x{2}  = test_x(:,test_y(:,2) ==1);
% mnist_x{3}  = test_x(:,test_y(:,3) ==1);
% mnist_x{4}  = test_x(:,test_y(:,4) ==1);
% mnist_x{5}  = test_x(:,test_y(:,5) ==1);
% mnist_x{6}  = test_x(:,test_y(:,6) ==1);
% mnist_x{7}  = test_x(:,test_y(:,7) ==1);
% mnist_x{8}  = test_x(:,test_y(:,8) ==1);
% mnist_x{9}  = test_x(:,test_y(:,9) ==1);
% mnist_x{10} = test_x(:,test_y(:,10)==1);
% 
% 
% save('./dataset/mnist_test_x.mat', 'mnist_x');
% 
% clear mnist_x;
% 
% train_x  = double(train_x')/ 255;
% train_y  = double(train_y);
% train_x = zscore(train_x);
%  
% mnist_x{1}  = train_x(:,train_y(:,1) ==1);
% mnist_x{2}  = train_x(:,train_y(:,2) ==1);
% mnist_x{3}  = train_x(:,train_y(:,3) ==1);
% mnist_x{4}  = train_x(:,train_y(:,4) ==1);
% mnist_x{5}  = train_x(:,train_y(:,5) ==1);
% mnist_x{6}  = train_x(:,train_y(:,6) ==1);
% mnist_x{7}  = train_x(:,train_y(:,7) ==1);
% mnist_x{8}  = train_x(:,train_y(:,8) ==1);
% mnist_x{9}  = train_x(:,train_y(:,9) ==1);
% mnist_x{10} = train_x(:,train_y(:,10)==1);
% 
% save('./dataset/mnist_train_x.mat', 'mnist_x');

load('./dataset/mnist_multimodal_84_16.mat');

%%Load dataset
train_x = imdb.image(:,imdb.set(1,:)==0);
train_y = imdb.label(imdb.set(1,:)==0, 1);

mnist_x{ 1} = train_x(:,find(train_y(:)==0));
mnist_x{ 2} = train_x(:,find(train_y(:)==1));
mnist_x{ 3} = train_x(:,find(train_y(:)==2));
mnist_x{ 4} = train_x(:,find(train_y(:)==3));
mnist_x{ 5} = train_x(:,find(train_y(:)==4));
mnist_x{ 6} = train_x(:,find(train_y(:)==5));
mnist_x{ 7} = train_x(:,find(train_y(:)==6));
mnist_x{ 8} = train_x(:,find(train_y(:)==7));
mnist_x{ 9} = train_x(:,find(train_y(:)==8));
mnist_x{10} = train_x(:,find(train_y(:)==9));

save('./dataset/mnist_mm_x.mat', 'mnist_x');




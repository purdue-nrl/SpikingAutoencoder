clear;
load('./output_final/AC_MNIST/one-image-per-audio/ac-epoch-20.mat');


for i = 1:1:20
   test_mse(i) = stats{i,2}.val.total_mse;
   train_mse(i) = stats{i,1}.train.total_mse;   
end

save('./output_final/AC_MNIST/one-image-per-audio/train_mse.mat', 'train_mse')
save('./output_final/AC_MNIST/one-image-per-audio/test_mse.mat', 'test_mse')
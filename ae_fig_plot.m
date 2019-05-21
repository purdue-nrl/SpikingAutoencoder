function [H] = ae_fig_plot(f, train_x, train_output_spikes, test_x, test_output_spikes, weights_code, weights_decode, train_loss, test_loss, train_mse, test_mse, x, output_dir)

figure(f);
subplot(2,4,1);
imagesc(reshape(train_x, 28,28)); colormap(gray); colorbar;
drawnow;
subplot(2,4,2);
imagesc(reshape(train_output_spikes, 28, 28)); colorbar; colormap(gray);
drawnow;
subplot(2,4,3);
imagesc(reshape(test_x, 28,28)); colormap(gray); colorbar;
drawnow;
subplot(2,4,4);
imagesc(reshape(test_output_spikes, 28, 28)); colorbar; colormap(gray);
drawnow;
subplot(2,4,5);
imagesc(weights_code);
colormap(gray);
colorbar;
title('Trained Weight-Code');
subplot(2,4,6);
imagesc(weights_decode);
colormap(gray);
colorbar;
drawnow;
title('Trained Weight-Decode');
subplot(2,4,7);
plot(x, train_loss, x , test_loss);
legend('train loss', 'test loss');  
subplot(2,4,8);
plot(x, train_mse, x , test_mse);
legend('train mse', 'test mse');
drawnow;


end
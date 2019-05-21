function [] = ac_fig_plot(f, im1, train_output_spikes, im2, test_output_spikes, avg_loss, avg_mse)
figure(f);
subplot(3,2,1);
imagesc(reshape(im1, 28,28)'); colormap(gray); colorbar;
drawnow;
subplot(3,2,2);
imagesc(reshape(train_output_spikes, 28, 28)'); colorbar; colormap(gray);
drawnow;
subplot(3,2,3);
imagesc(reshape(im2, 28,28)'); colormap(gray); colorbar;
drawnow;
subplot(3,2,4);
imagesc(reshape(test_output_spikes, 28, 28)'); colorbar; colormap(gray);
drawnow;
subplot(3,2,5);
plot(avg_loss);
drawnow;
subplot(3,2,6);
plot(avg_mse);
drawnow;

end
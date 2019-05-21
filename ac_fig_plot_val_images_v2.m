function [] = ac_fig_plot_val_images_v2(f, im1, output_spikes)
figure(f);
subplot(1,2,1);
imagesc(reshape(im1, 28,28)'); colormap(gray); colorbar;
drawnow;
subplot(1,2,2);
imagesc(reshape(output_spikes, 28, 28)'); colorbar; colormap(gray);
drawnow;
% subplot(2,2,3);
% imagesc(reshape(im2, 28,28)'); colormap(gray); colorbar;
% drawnow;
%subplot(2,2,4);
%plot(avg_mse);
drawnow;

end
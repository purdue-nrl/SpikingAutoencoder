% function to train spike auto-encoder

function [ae, loss, ae_mse] = ae_ann_train(ae, input, opts)
    
output = zeros(ae.output_size, opts.batch_size);

s = 1;
%figure(5);
%imagesc(reshape(input(:,s), 28,28)'); colormap(gray);
ae = ae.code(input);
ae = ae.decode();
output = ae.output.a;
ae = ae.calculate_loss(input);
ae = ae.calculate_gradients(opts);
ae = ae.apply_gradients(opts);
% figure(2);
% subplot(2,2,1);
% imagesc(ae.weights_code); colorbar; colormap(gray);
% subplot(2,2,2);
% imagesc(ae.weights_decode); colorbar; colormap(gray);
% subplot(2,2,3);
% imagesc(ae.hidden.cache.dL_dW); colorbar; colormap(gray);
% subplot(2,2,4);
% imagesc(ae.output.cache.dL_dW); colorbar; colormap(gray);
% figure(3);
% subplot(1,3,1)
% imagesc(reshape(input(:,s), 28, 28)'); colorbar; colormap(gray);
% subplot(1,3,2)
% imagesc(reshape(ae.hidden.a(:,s), sqrt(ae.hidden_size), sqrt(ae.hidden_size))'); colorbar; colormap(gray);
% subplot(1,3,3)
% imagesc(reshape(ae.output.a(:,s), 28, 28)'); colorbar; colormap(gray);
% drawnow;

input = zscore(input);
output = zscore(output);
err = input - output;
loss = 0.5*sum(err.^2,1);
loss = sum(loss,2);
ae_mse = mse(input, output);

%         fprintf(' complete Loss = %f \n', loss(round,t));
%         output_spikes = zeros(size(ae.output.spikes));
%         ae = ae.initialize(opts);
%         for t = 1:10
%             ae = ae.code(spike_input(:,:,t), opts);
%             ae = ae.decode(opts);
%             output_spikes = bitor(output_spikes,ae.output.spikes);
%             figure(4);
%             imagesc(reshape(output_spikes(:,s), 28, 28)'); colorbar; colormap(gray);
%             %ae = ae.initialize(opts);
%         end



   
end

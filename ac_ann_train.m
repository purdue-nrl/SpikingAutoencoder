% function to train spike auto-encoder

function [ac, loss, ac_mse] = ac_ann_train(ac, ae, input, target, opts)
    
%s = 1;
%figure(5);
%imagesc(reshape(input(:,s), 28,28)'); colormap(gray);
ac = ac.code(input);
ac = ac.decode();
ae = ae.code(target);
if opts.quantize
    h = imquantize(ae.hidden.a, opts.levels);    
    h = (h-1)*(opts.levels(1,2)-opts.levels(1,1));
else
    h = ae.hidden.a;
end

ac = ac.calculate_loss(h);
ac = ac.calculate_gradients(opts);
ac = ac.apply_gradients(opts);

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


err = h - ac.output.a;
loss = 0.5*sum(err.^2,1);
loss = sum(loss,2);
% reconstruction loss
ae.hidden.a = ac.output.a;
ae = ae.decode();
output = zscore(ae.output.a);
target = zscore(target);
ac_mse = mse(target, output);

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

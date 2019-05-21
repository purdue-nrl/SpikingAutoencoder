% function to train spike auto-encoder

function [ae, loss, ae_mse, ae_mse_pixel] = ae_train(ae, input, opts)
    
spike_input = pixel_to_spike(input, opts.dt, opts.duration, opts.max_rate);
output = zeros(ae.output_size, opts.batch_size);

s = 1;
%figure(5);
%imagesc(reshape(input(:,s), 28,28)'); colormap(gray);
sparse = [];
ae = ae.initialize(opts);
for t = 1:size(opts.dt:opts.dt:opts.duration,2)
    ae = ae.code(spike_input(:,:,t), opts);
    %spike_activity.hidden(:,:,t) = ae.hidden.spikes;
    ae = ae.decode(opts);
    output = output + ae.output.spikes;
    ae = ae.calculate_loss(spike_input(:,:,t), opts);
    %ae = ae.calculate_loss(any(spike_input, 3), opts);
    %sparse(t) = ae.sparse;
    ae = ae.calculate_gradients(opts, t);
    ae = ae.apply_gradients(opts, 1, t);
%     figure(2);
%     subplot(2,2,1);
%     imagesc(ae.weights_code); colorbar; colormap(gray);
%     subplot(2,2,2);
%     imagesc(ae.weights_decode); colorbar; colormap(gray);
%     subplot(2,2,3);
%     imagesc(ae.hidden.cache.dL_dW); colorbar; colormap(gray);
%     subplot(2,2,4);
%     imagesc(ae.output.cache.dL_dW); colorbar; colormap(gray);
%     figure(3);
%     subplot(2,3,1)
%     imagesc(reshape(spike_input(:,s,t), 28, 28)'); colorbar; colormap(gray);
%     subplot(2,3,4)
%     imagesc(reshape(input(:,s), 28, 28)'); colorbar; colormap(gray);
%     subplot(2,3,2)
%     imagesc(reshape(ae.hidden.spikes(:,s), sqrt(ae.hidden_size), sqrt(ae.hidden_size))'); colorbar; colormap(gray);
%     subplot(2,3,5)
%     imagesc(reshape(ae.hidden.cache.v_mem(:,s), sqrt(ae.hidden_size), sqrt(ae.hidden_size))'); colorbar; colormap(gray);
%     subplot(2,3,3)
%     imagesc(reshape(ae.output.spikes(:,s), 28, 28)'); colorbar; colormap(gray);
%     subplot(2,3,6)
%     imagesc(reshape(ae.output.cache.v_mem(:,s), 28, 28)'); colorbar; colormap(gray);
%     drawnow;
    %fprintf('Loss: %f, i: %1.0f h: %1.0f o: %1.0f \n', ae.Loss(:,s), numel(find(spike_input(:,s,t)>0)), numel(find(ae.hidden.spikes(:,s)>0)),numel(find(ae.output.spikes(:,s)>0)) );
    ae = ae.clearcache();
end
%output =  output/max(output);
output2 = zscore(output);
input2 = sum(spike_input,3);
%input2 = input2/max(input2);
input2 = zscore(input2);
err = input2 - output2;
loss = 0.5*sum(err.^2,1);
loss = sum(loss,2);
ae_mse = mse(input2, output2);
ae_mse_pixel = mse(zscore(input), output2);

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

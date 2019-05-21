function [ae, loss, ae_mse, ae_mse_pixel] = ae_test(ae, input, opts)

    spike_input = pixel_to_spike(input, opts.dt, opts.duration, opts.max_rate);
    output = zeros(size(input));
    
    for t = 1:size(opts.dt:opts.dt:opts.duration,2)
        ae = ae.code(spike_input(:,:,t), opts);
        ae = ae.decode(opts);
        output = output + ae.output.spikes;
        ae = ae.calculate_loss(spike_input(:,:,t), opts);        
    end
    %output =  output/max(output);
    output2 = zscore(output);
    input2 = sum(spike_input,3);
    %input2 = input2/max(input2);
    input2 = zscore(input2);
    err = input2 - output2;
    final_loss = 0.5*sum(err.^2,1);
    loss = sum(final_loss,2);
    ae_mse = mse(input2, output2);
    ae_mse_pixel = mse(zscore(input), output2);

end
function ac = ac_ff_test(ac, spike_input, opts)
% Get input impulse from incoming spikes
z = ac.layers{1}.weights*spike_input;
% Add input to membrane potential and compute gradients
ac.layers{1}.dz_dW = spike_input';
switch (opts.neuron_model)
    case 'IF'
        ac.layers{1}.v_mem = min(ac.layers{1}.v_mem + z, opts.threshold);
        
    case 'LIF'
        ac.layers{1}.v_mem = min(ac.layers{1}.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
        
        
end
% Check for spiking
ac.layers{1}.spikes = ac.layers{1}.v_mem >= opts.threshold;
% Reset
ac.layers{1}.v_mem_prev = ac.layers{1}.v_mem;
ac.layers{1}.v_mem(ac.layers{1}.spikes) = 0;
clear z;
z = ac.layers{2}.weights*ac.layers{1}.spikes;
% Add input to membrane potential
ac.layers{2}.dz_dW = ac.layers{1}.spikes';
switch (opts.neuron_model)
    case 'IF'
        ac.layers{2}.v_mem = min(ac.layers{2}.v_mem + z, opts.threshold);
                
    case 'LIF'
        ac.layers{2}.v_mem = min(ac.layers{2}.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
        
end
% Check for spiking
ac.layers{2}.spikes = ac.layers{2}.v_mem >= opts.threshold;
% Store v_mem in cache for back propagation before resetting
ac.layers{2}.v_mem_prev = ac.layers{2}.v_mem;
% Reset
ac.layers{2}.v_mem(ac.layers{2}.spikes) = 0;

end
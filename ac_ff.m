function ac = ac_ff(ac, spike_input, opts)
% Get input impulse from incoming spikes
z = ac.layers{1}.weights*spike_input;
% Add input to membrane potential and compute gradients
ac.layers{1}.dz_dW = spike_input';
switch (opts.neuron_model)
    case 'IF'
        ac.layers{1}.v_mem = min(ac.layers{1}.v_mem + z, opts.threshold);
        x = ac.layers{1}.v_mem - opts.threshold;
        ac.layers{1}.da_dvmem = (exp(-x)./((1+exp(-x)).^2));
        ac.layers{1}.dvmem_dW = ac.layers{1}.dvmem_dW + ac.layers{1}.dz_dW;
        ac.layers{1}.dvmem_dW_prev = ac.layers{1}.dvmem_dW;
    case 'LIF'
        ac.layers{1}.v_mem = min(ac.layers{1}.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
        x = ac.layers{1}.v_mem - opts.threshold;
        ac.layers{1}.da_dvmem = (exp(-x)./((1+exp(-x)).^2));
        ac.layers{1}.dvmem_dW = ac.layers{1}.dvmem_dW_prev*(1-opts.dt/opts.tau) + ac.layers{1}.dz_dW;
        ac.layers{1}.dvmem_dW_prev = ac.layers{1}.dvmem_dW;
        
end
% Check for spiking
ac.layers{1}.spikes = ac.layers{1}.v_mem >= opts.threshold;
% Reset
ac.layers{1}.v_mem_prev = ac.layers{1}.v_mem;
ac.layers{1}.v_mem(ac.layers{1}.spikes) = 0;
ac.layers{1}.dvmem_dW_prev = (1 - sum(ac.layers{1}.spikes,1)/size(ac.layers{1}.weights,2))'.*ac.layers{1}.dvmem_dW_prev;
clear z;
z = ac.layers{2}.weights*ac.layers{1}.spikes;
% Add input to membrane potential
ac.layers{2}.dz_dW = ac.layers{1}.spikes';
switch (opts.neuron_model)
    case 'IF'
        ac.layers{2}.v_mem = min(ac.layers{2}.v_mem + z, opts.threshold);
        ac.layers{2}.dvmem_dW = ac.layers{2}.dvmem_dW_prev + ac.layers{2}.dz_dW;
        ac.layers{2}.dvmem_dW_prev = ac.layers{2}.dvmem_dW;
        
    case 'LIF'
        ac.layers{2}.v_mem = min(ac.layers{2}.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
        ac.layers{2}.dvmem_dW = ac.layers{2}.dvmem_dW_prev*(1-opts.dt/opts.tau) + ac.layers{2}.dz_dW;
        ac.layers{2}.dvmem_dW_prev = ac.layers{2}.dvmem_dW;
end
% Check for spiking
ac.layers{2}.spikes = ac.layers{2}.v_mem >= opts.threshold;
% Store v_mem in cache for back propagation before resetting
ac.layers{2}.v_mem_prev = ac.layers{2}.v_mem;
% Reset
ac.layers{2}.v_mem(ac.layers{2}.spikes) = 0;
ac.layers{2}.dvmem_dW_prev = (1 - sum(ac.layers{2}.spikes,1)/size(ac.layers{2}.weights,2))'.*ac.layers{2}.dvmem_dW_prev;
end
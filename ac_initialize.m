function ac = ac_initialize(ac, opts)

    ac.error = [];
    ac.mse = [];
    ac.layers{1}.v_mem = zeros(size(ac.layers{1}.weights,1), opts.batch_size);
    ac.layers{1}.v_mem_prev = zeros(size(ac.layers{1}.weights,1), opts.batch_size);
    ac.layers{1}.spikes = zeros(size(ac.layers{1}.weights,1), opts.batch_size);
    ac.layers{1}.dL_dW =  zeros(size(ac.layers{1}.weights));          
    ac.layers{1}.m1 = zeros(size(ac.layers{1}.weights));      
    ac.layers{1}.m2 = zeros(size(ac.layers{1}.weights)); 
    ac.layers{1}.da_dvmem = zeros(size(ac.layers{1}.weights,1), opts.batch_size);
    ac.layers{1}.dvmem_dW = zeros(opts.batch_size,size(ac.layers{1}.weights,2));
    ac.layers{1}.dvmem_dW_prev = zeros(opts.batch_size,size(ac.layers{1}.weights,2));      
            
    ac.layers{2}.v_mem = zeros(size(ac.layers{2}.weights,1), opts.batch_size);
    ac.layers{2}.spikes = zeros(size(ac.layers{2}.weights,1), opts.batch_size);
    ac.layers{2}.v_mem_prev = zeros(size(ac.layers{2}.weights,1), opts.batch_size);
    ac.layers{2}.dL_dW = zeros(size(ac.layers{2}.weights));
    ac.layers{2}.m1 = zeros(size(ac.layers{2}.weights));
    ac.layers{2}.m2 = zeros(size(ac.layers{2}.weights));
    ac.layers{2}.dvmem_dW = zeros(opts.batch_size, size(ac.layers{2}.weights,2));
    ac.layers{2}.dvmem_dW_prev = zeros(opts.batch_size, size(ac.layers{2}.weights,2)); 
    ac.layers{2}.dz_dW = zeros(opts.batch_size, size(ac.layers{2}.weights,2));
    
    

end
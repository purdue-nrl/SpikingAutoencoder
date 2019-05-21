classdef auto_encoder < handle
    % object that defines a singe auto_encoder
    
    properties
        %basic property defining number of units
        input_size;
        output_size;
        hidden_size;
        sparse;
        % weight matrix for coding
        weights_code = [];
        % weight matrix for decoding
        weights_decode = [];
        % structures that contain neuron properties
        % input
        input = struct( 'spikes'    , []);
        % hidden: properties of the hidden neuron group
        hidden = struct(    'v_mem'         , [], ...
                            'spikes'        , [], ...
                            'cache', struct('v_mem', [], 'da_dvmem', [], 'dz_dW', [], ...
                                            'dL_dW', [], 'm1', [], 'm2', []));
        % output: properties of the output neuron group
        output = struct(    'v_mem'         , [], ...
                            'spikes'        , [], ...
                            'cache', struct('v_mem', [], 'dL_dvmem', [], 'dz_dW', [], ...
                                            'dL_dW', [], 'm1', [], 'm2', []));
        % performance parameters
        
        error       = [];
        batch_loss  = [];
        avg_loss    = [];
        batch_mse   = [];
        avg_mse     = [];
        batch_mse_pixel = [];
                    

    end
    
    methods
        function ae = auto_encoder(input_size, hidden_size, output_size)
            
            ae.input_size = input_size;
            ae.output_size = output_size;
            ae.hidden_size = hidden_size;
            ae.weights_code = (rand(hidden_size, input_size) - 0.5) * 0.1 * 2;
            ae.weights_decode = (rand(output_size, hidden_size) - 0.5) * 0.1 * 2;
                        
        end
        
        function ae = initialize(ae, opts)
            
            ae.hidden.v_mem = zeros(ae.hidden_size, opts.batch_size);
            ae.hidden.spikes = zeros(ae.hidden_size, opts.batch_size);
            ae.hidden.cache.v_mem = zeros(ae.hidden_size, opts.batch_size);
            ae.hidden.cache.dL_dW = zeros(size(ae.weights_code));
            ae.hidden.cache.m1 = zeros(size(ae.weights_code));
            ae.hidden.cache.m2 = zeros(size(ae.weights_code));
            ae.hidden.cache.dvmem_dW_prev = zeros(opts.batch_size, ae.input_size);
            
            ae.output.v_mem = zeros(ae.output_size, opts.batch_size); 
            ae.output.spikes = zeros(ae.output_size, opts.batch_size); 
            ae.output.cache.dL_dW = zeros(size(ae.weights_decode));
            ae.output.cache.m1 = zeros(size(ae.weights_decode));
            ae.output.cache.m2 = zeros(size(ae.weights_decode));
            ae.output.cache.dvmem_dW_prev = zeros(opts.batch_size, ae.hidden_size);
            ae.output.cache.dz_dW = zeros(opts.batch_size, ae.hidden_size);
            ae.output.cache.v_mem = zeros(ae.output_size, opts.batch_size);
        end
        
        function ae = clearcache(ae)
            ae.output.cache.dL_dW = zeros(size(ae.weights_decode));
            ae.hidden.cache.dL_dW = zeros(size(ae.weights_code));
        end
        
        function ae = code(ae, spike_input, opts)            
            ae.input.spikes = spike_input;
            % Get input impulse from incoming spikes
            z = ae.weights_code*ae.input.spikes;
            % Add input to membrane potential and compute gradients
            ae.hidden.cache.dz_dW = spike_input';
            switch (opts.neuron_model)
                case 'IF' 
                    ae.hidden.v_mem = min(ae.hidden.v_mem + z, opts.threshold);
                    x = ae.hidden.v_mem - opts.threshold;
                    ae.hidden.cache.da_dvmem = (exp(-x)./((1+exp(-x)).^2));
                    ae.hidden.cache.dvmem_dW = ae.hidden.cache.dvmem_dW_prev + ae.hidden.cache.dz_dW;
                    ae.hidden.cache.dvmem_dW_prev = ae.hidden.cache.dvmem_dW;
                case 'LIF'
                    ae.hidden.v_mem = min(ae.hidden.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
                    x = ae.hidden.v_mem - opts.threshold;
                    ae.hidden.cache.da_dvmem = (exp(-x)./((1+exp(-x)).^2));
                    ae.hidden.cache.dvmem_dW = ae.hidden.cache.dvmem_dW_prev*(1-opts.dt/opts.tau) + ae.hidden.cache.dz_dW;
                    ae.hidden.cache.dvmem_dW_prev = ae.hidden.cache.dvmem_dW;  
            end
            % Check for spiking
            ae.hidden.spikes = ae.hidden.v_mem >= opts.threshold;            
            % Reset
            ae.hidden.cache.v_mem = ae.hidden.v_mem;
            ae.hidden.v_mem(ae.hidden.spikes) = 0;
            ae.hidden.cache.dvmem_dW_prev = (1 - sum(ae.hidden.spikes,1)/784)'.*ae.hidden.cache.dvmem_dW_prev;

        end
        
        function ae = decode(ae, opts)    
            % Get input impulse from incoming spikes
            z = ae.weights_decode*ae.hidden.spikes;
            % Add input to membrane potential
            ae.output.cache.dz_dW = ae.hidden.spikes';              
            switch (opts.neuron_model)
                case 'IF' 
                    ae.output.v_mem = min(ae.output.v_mem + z, opts.threshold);
                    ae.output.cache.dvmem_dW = ae.output.cache.dvmem_dW_prev + ae.output.cache.dz_dW;
                    ae.output.cache.dvmem_dW_prev = ae.output.cache.dvmem_dW;
                    
                case 'LIF'
                    ae.output.v_mem = min(ae.output.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
                    ae.output.cache.dvmem_dW = ae.output.cache.dvmem_dW_prev*(1-opts.dt/opts.tau) + ae.output.cache.dz_dW;
                    ae.output.cache.dvmem_dW_prev = ae.output.cache.dvmem_dW;                    
            end            
            % Check for spiking
            ae.output.spikes = ae.output.v_mem >= opts.threshold;
            % Store v_mem in cache for back propagation before resetting 
            ae.output.cache.v_mem = ae.output.v_mem;
            % Reset
            ae.output.v_mem(ae.output.spikes) = 0;
            ae.output.cache.dvmem_dW_prev = (1 - sum(ae.output.spikes,1)/784)'.*ae.output.cache.dvmem_dW_prev;            
         
        end
        
        function ae = code_test(ae, spike_input, opts)
            ae.input.spikes = spike_input;
            % Get input impulse from incoming spikes
            z = ae.weights_code*ae.input.spikes;
            % Add input to membrane potential and compute gradients
            switch (opts.neuron_model)
                case 'IF'
                    ae.hidden.v_mem = min(ae.hidden.v_mem + z, opts.threshold);
                case 'LIF'
                    ae.hidden.v_mem = min(ae.hidden.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
            end
            % Check for spiking
            ae.hidden.spikes = ae.hidden.v_mem >= opts.threshold;
            % Reset
            ae.hidden.v_mem(ae.hidden.spikes) = 0;
            
        end
        
        
        
        function ae = decode_test(ae, opts)
            % Get input impulse from incoming spikes
            z = ae.weights_decode*ae.hidden.spikes;
            % Add input to membrane potential
            switch (opts.neuron_model)
                case 'IF'
                    ae.output.v_mem = min(ae.output.v_mem + z, opts.threshold);
                case 'LIF'
                    ae.output.v_mem = min(ae.output.v_mem*(1-opts.dt/opts.tau) + z, opts.threshold);
            end
            % Check for spiking
            ae.output.spikes = ae.output.v_mem >= opts.threshold;
            % Reset
            ae.output.v_mem(ae.output.spikes) = 0;
        end
        
        
        function ae = calculate_loss(ae, target_spikes, opts)
            
            target_v_mem = target_spikes*opts.threshold*opts.scale;%desired membrane
            switch opts.mask
                case 'bitxor'
                    mask = bitxor(target_spikes,ae.output.spikes);
                case 'bitor'
                    mask = bitxor(target_spikes,ae.output.spikes);
                case 'none'
                    mask = ones(size(ae.output.spikes));
                otherwise
                    mask = ones(size(ae.output.spikes));
            end
            ae.error = (target_v_mem - ae.output.cache.v_mem).*mask; 
            ae.sparse = nnz(ae.error)/(size(ae.error,1)*size(ae.error,2));
            % calculate the mean squared error
            % no regularization term used
            %loss = 0.5*sum(ae.error.^2,1);
            % calculate average loss over batch
           
        end
        
        function ae = calculate_gradients(ae, opts, t)
            %calculate gradients in decode layer
            ae.output.cache.dL_dvmem = -ae.error; 
            ae.output.cache.dL_dW = ae.output.cache.dL_dW + ae.output.cache.dL_dvmem*ae.output.cache.dvmem_dW*(1/opts.batch_size);            
            % calculate gradient in code layer                        
            ae.hidden.cache.dL_dW = ae.hidden.cache.dL_dW + ((ae.weights_decode'*ae.output.cache.dL_dvmem).*ae.hidden.cache.da_dvmem)*ae.hidden.cache.dvmem_dW*(1/opts.batch_size);
            % gradient clipping
            if (opts.grad_clip)
                N_op = norm(ae.output.cache.dL_dW); %calculate L2 norm
                if N_op > opts.grad_clip_thresh
                    ae.output.cache.dL_dW = opts.grad_clip_thresh*ae.output.cache.dL_dW/N_op;
                end
                H_op = norm(ae.hidden.cache.dL_dW);
                if H_op > opts.grad_clip_thresh
                    ae.hidden.cache.dL_dW = opts.grad_clip_thresh*ae.hidden.cache.dL_dW/H_op;
                end
            end            
            % adam optimization
            if (opts.adam)
                ae.hidden.cache.m1 = (opts.beta1*ae.hidden.cache.m1 + (1-opts.beta1)*ae.hidden.cache.dL_dW);
                ae.hidden.cache.m2 = (opts.beta2*ae.hidden.cache.m2 + (1-opts.beta2)*(ae.hidden.cache.dL_dW.^2));
                ae.output.cache.m1 = (opts.beta1*ae.output.cache.m1 + (1-opts.beta1)*ae.output.cache.dL_dW);
                ae.output.cache.m2 = (opts.beta2*ae.output.cache.m2 + (1-opts.beta2)*(ae.output.cache.dL_dW.^2));
            end
            
        end
        
        function ae = apply_gradients(ae, opts, timesteps, t)
            if (opts.adam)
                m1_code = ae.hidden.cache.m1/(1 - opts.beta1^t);
                m2_code = ae.hidden.cache.m2/(1 - opts.beta2^t);
                m1_decode = ae.output.cache.m1/(1 - opts.beta1^t);
                m2_decode = ae.output.cache.m2/(1 - opts.beta2^t);
                ae.weights_code = ae.weights_code - opts.alpha*(m1_code./(m2_code.^0.5 + opts.epsilon));
                ae.weights_decode = ae.weights_decode - opts.alpha*(m1_decode./(m2_decode.^0.5 + opts.epsilon));
            else                
                ae.weights_code = ae.weights_code*(1-opts.weight_decay) - opts.alpha*ae.hidden.cache.dL_dW*(1/timesteps);
                ae.weights_decode = ae.weights_decode*(1-opts.weight_decay) - opts.alpha*ae.output.cache.dL_dW*(1/timesteps);
            end
        end
        
    end
    
end
    
            
            
            

            
    
    
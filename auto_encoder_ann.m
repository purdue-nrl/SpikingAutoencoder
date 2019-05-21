classdef auto_encoder_ann < handle
    % object that defines a singe auto_encoder
    
    properties
        %basic property defining number of units
        input_size;
        output_size;
        hidden_size;
        % weight matrix for coding
        weights_code = [];
        % weight matrix for decoding
        weights_decode = [];
        % structures that contain neuron properties
        % hidden: properties of the hidden neuron group
        hidden = struct(    'a'         , [], ...
                            'z'        , [], ...
                            'cache', struct('da_dz', [], 'dz_dW', [], ...
                                            'dL_dW', [], 'm1', [], 'm2', []));
        % output: properties of the output neuron group
        output = struct(    'a'         , [], ...
                            'z'        , [], ...
                            'cache', struct('da_dz', [], 'dz_dW', [], ...
                                            'dL_dW', [], 'm1', [], 'm2', []));
        % performance parameters
        
        error       = [];
        batch_loss  = [];
        avg_loss    = [];
        batch_mse   = [];
        avg_mse     = [];
        
                    

    end
    
    methods
        function ae = auto_encoder_ann(input_size, hidden_size, output_size)
            
            ae.input_size = input_size;
            ae.output_size = output_size;
            ae.hidden_size = hidden_size;
            ae.weights_code = (rand(hidden_size, input_size) - 0.5) * 0.01 * 2;
            ae.weights_decode = (rand(output_size, hidden_size) - 0.5) * 0.01 * 2;
                        
        end
        
        function ae = initialize(ae, opts)
            
            ae.hidden.cache.dL_dW = zeros(size(ae.weights_code));
            ae.hidden.cache.m1 = zeros(size(ae.weights_code));
            ae.hidden.cache.m2 = zeros(size(ae.weights_code));
                        
            ae.output.cache.dL_dW = zeros(size(ae.weights_decode));
            ae.output.cache.m1 = zeros(size(ae.weights_decode));
            ae.output.cache.m2 = zeros(size(ae.weights_decode));
            ae.output.cache.dz_dW = zeros(opts.batch_size, ae.hidden_size);
        end
        
       
        function ae = code(ae, input)            
            % Get input to neuron
            ae.hidden.z = ae.weights_code*input;
            % Add input to membrane potential and compute gradients
            ae.hidden.cache.dz_dW = input';
            ae.hidden.a = max(0, ae.hidden.z); %%ReLU activation 
            ae.hidden.cache.da_dz = ae.hidden.z > 0;

        end
        
        function ae = decode(ae)    
            % Get input impulse from incoming spikes
            ae.output.z = ae.weights_decode*ae.hidden.a;
            % Add input to membrane potential
            ae.output.cache.dz_dW = ae.hidden.a';              
            ae.output.a = max(0, ae.output.z); %%ReLU activation                      
            ae.output.cache.da_dz = ae.output.z > 0;
        end
        
        function ae = code_test(ae, input)
	    % Get input to neuron
            ae.hidden.z = ae.weights_code*input;
            % Add input to membrane potential and compute gradients
            ae.hidden.a = max(0, ae.hidden.z); %%ReLU activation     
        end
        
        
        
        function ae = decode_test(ae)
            % Get input impulse from incoming spikes
            ae.output.z = ae.weights_decode*ae.hidden.a;             
            ae.output.a = max(0, ae.output.z); %%ReLU activation 
        end
        
        
        function ae = calculate_loss(ae, target)
            
            ae.error = (target - ae.output.a);        
           
        end
        
        function ae = calculate_gradients(ae, opts)
            %calculate gradients in decode layer
            ae.output.cache.dL_da = -ae.error; 
            ae.output.cache.dL_dz = ae.output.cache.dL_da.*ae.output.cache.da_dz;
            ae.output.cache.dL_dW = ae.output.cache.dL_dz*ae.output.cache.dz_dW*(1/opts.batch_size);   
            % calculate gradient in code layer 
            ae.hidden.cache.dL_dz = (ae.weights_decode'*ae.output.cache.dL_dz).*ae.hidden.cache.da_dz;
            ae.hidden.cache.dL_dW = ae.hidden.cache.dL_dz*ae.hidden.cache.dz_dW*(1/opts.batch_size);
            % gradient clipping
            if (opts.adam)
                ae.hidden.cache.m1 = (opts.beta1*ae.hidden.cache.m1 + (1-opts.beta1)*ae.hidden.cache.dL_dW);
                ae.hidden.cache.m2 = (opts.beta2*ae.hidden.cache.m2 + (1-opts.beta2)*(ae.hidden.cache.dL_dW.^2));
                ae.output.cache.m1 = (opts.beta1*ae.output.cache.m1 + (1-opts.beta1)*ae.output.cache.dL_dW);
                ae.output.cache.m2 = (opts.beta2*ae.output.cache.m2 + (1-opts.beta2)*(ae.output.cache.dL_dW.^2));
            end
            
        end
        
        function ae = apply_gradients(ae, opts)
            if (opts.adam)
                t = opts.batch_number;
                m1_code = ae.hidden.cache.m1/(1 - opts.beta1^t);
                m2_code = ae.hidden.cache.m2/(1 - opts.beta2^t);
                m1_decode = ae.output.cache.m1/(1 - opts.beta1^t);
                m2_decode = ae.output.cache.m2/(1 - opts.beta2^t);
                ae.weights_code = ae.weights_code - opts.alpha*(m1_code./(m2_code.^0.5 + opts.epsilon));
                ae.weights_decode = ae.weights_decode - opts.alpha*(m1_decode./(m2_decode.^0.5 + opts.epsilon));
            else                
                ae.weights_code = ae.weights_code*(1-opts.weight_decay) - opts.alpha*ae.hidden.cache.dL_dW;
                ae.weights_decode = ae.weights_decode*(1-opts.weight_decay) - opts.alpha*ae.output.cache.dL_dW;
            end
            
        end
        
    end
    
end
    
            
            
            

            
    
    

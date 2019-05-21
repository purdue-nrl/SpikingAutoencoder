function ac = ac_calcgrad(ac, opts, t)
%calculate gradients in layer-2
ac.layers{2}.dL_dW = (-ac.error)*ac.layers{2}.dvmem_dW*(1/opts.batch_size);
% calculate gradient in layer-1
ac.layers{1}.dL_dW = ((ac.layers{2}.weights'*(-ac.error)).*ac.layers{1}.da_dvmem)*ac.layers{1}.dvmem_dW*(1/opts.batch_size);
% gradient clipping
if (opts.grad_clip)
    N_op = norm(ac.layers{2}.dL_dW); %calculate L2 norm
    if N_op > opts.grad_clip_thresh
        ac.layers{2}.dL_dW = opts.grad_clip_thresh*ac.layers{2}.dL_dW/N_op;
    end
    H_op = norm(ac.layers{1}.dL_dW);
    if H_op > opts.grad_clip_thresh
        ac.layers{1}.dL_dW = opts.grad_clip_thresh*ac.layers{1}.dL_dW/H_op;
    end
end
% adam optimization
if (opts.adam)
    ac.layers{1}.m1 = (opts.beta1*ac.layers{1}.m1 + (1-opts.beta1)*ac.layers{1}.dL_dW);
    ac.layers{1}.m2 = (opts.beta2*ac.layers{1}.m2 + (1-opts.beta2)*(ac.layers{1}.dL_dW.^2));
    ac.layers{2}.m1 = (opts.beta1*ac.layers{2}.m1 + (1-opts.beta1)*ac.layers{2}.dL_dW);
    ac.layers{2}.m2 = (opts.beta2*ac.layers{2}.m2 + (1-opts.beta2)*(ac.layers{2}.dL_dW.^2));
end

if (opts.adam)
    m1_code = ac.layers{1}.m1/(1 - opts.beta1^t);
    m2_code = ac.layers{1}.m2/(1 - opts.beta2^t);
    m1_decode = ac.layers{2}.m1/(1 - opts.beta1^t);
    m2_decode = ac.layers{2}.m2/(1 - opts.beta2^t);
    ac.layers{1}.weights = ac.layers{1}.weights - opts.alpha*(m1_code./(m2_code.^0.5 + opts.epsilon));
    ac.layers{2}.weights = ac.layers{2}.weights - opts.alpha*(m1_decode./(m2_decode.^0.5 + opts.epsilon));
else
    ac.layers{1}.weights = ac.layers{1}.weights*(1-opts.weight_decay) - opts.alpha*ac.layers{1}.dL_dW;
    ac.layers{2}.weights = ac.layers{2}.weights*(1-opts.weight_decay) - opts.alpha*ac.layers{2}.dL_dW;



end
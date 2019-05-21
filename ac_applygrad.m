function ac = ac_applygrad(ac, opts, t)
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
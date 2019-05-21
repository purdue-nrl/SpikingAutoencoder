function [ac, loss, ac_mse] = ac_ann_test_v2(ac, ae, input, target)
    load mnist_mm_x.mat;
    ac = ac.code(input);
    ac = ac.decode();
    ae.hidden.a = ac.output.a;
    ae = ae.decode();
    % reconctruction loss mse
    output = ae.output.a;
    output = zscore(output);
    target = zscore(target);
    
    ac_mse = mse(target, output);
    
    % loss over hidden state mismatch
    ae = ae.code(target);
    err = ae.hidden.a - ac.output.a;
    final_loss = 0.5*sum(err.^2,1);
    loss = sum(final_loss,2);
    

end
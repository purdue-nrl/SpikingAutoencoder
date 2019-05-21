function [ae, loss, ae_mse] = ae_ann_test(ae, input, opts)

    output = zeros(size(input));
    ae = ae.code(input);
    ae = ae.decode();
    output = ae.output.a;
    ae = ae.calculate_loss(input);        
    output = zscore(output);
    input = zscore(input);
    err = input - output;
    final_loss = 0.5*sum(err.^2,1);
    loss = sum(final_loss,2);
    ae_mse = mse(input, output);

end
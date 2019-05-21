function [ac, loss, ac_mse] = ac_ann_test(ac, ae, input, target, label)

% compares the reconstructed image to a subset of 100 train images having
% the same label and selects the min MSE.

    load mnist_mm_x.mat;
    ac = ac.code(input);
    ac = ac.decode();
    ae.hidden.a = ac.output.a;
    ae = ae.decode();
    % reconctruction loss mse
    output = ae.output.a;
    output = zscore(output);
    target = zscore(target);
    
    %ac_mse = mse(target, output);
    
    % loss over hidden state mismatch
    ae = ae.code(target);
    err = ae.hidden.a - ac.output.a;
    final_loss = 0.5*sum(err.^2,1);
    loss = sum(final_loss,2);
    
    ac_mse_temp = zeros(1,size(output,2));
    %best_image = zeros(784. size(output,2));
    for k = 1:size(output,2)
       [ac_mse_temp(1, k), ~] = min_mse(output(:,k), label(k), mnist_x);         
    end 
    ac_mse = sum(ac_mse_temp, 2)/size(input,2);


end

function [ae_mse, image] = min_mse(x, label, mnist_x)
    y = mnist_x{label+1};
    %calc_mse = zeros(1,100);
    calc_mse = ((y - x).^2);
    calc_mse = sum(calc_mse, 1)/784;
    
    %for i = 1:size(y,2)
    %    calc_mse(i) = mse(x,y(:,i));
    %end
    [ae_mse, j] = min(calc_mse);
    image = mnist_x{label+1}(:,j);
end
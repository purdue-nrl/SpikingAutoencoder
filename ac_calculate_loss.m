function [ac, loss] = ac_calculate_loss(ac, y, opts)
target_v_mem = y*opts.threshold*opts.scale;%desired membrane
mask = bitxor(y,ac.layers{2}.spikes);
ac.error = (target_v_mem - ac.layers{2}.v_mem_prev).*mask;
% calculate the mean squared error
loss = 0.5*sum(ac.error.^2,1);
loss = sum(loss,2);
end
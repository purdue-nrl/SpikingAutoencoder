function [ac, ev_stats] = ac_train(ac, ae, imdb, opts)
    load mnist_mm_x.mat;
    if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
    if isempty(opts.train), opts.train = find(imdb.set==0) ; end
    if isempty(opts.val), opts.val = find(imdb.set==1) ; end
    if isempty(opts.val), opts.val = find(imdb.set==2) ; end
    if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
        opts.train = [] ;
    end
    if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
        opts.val = [] ;
    end
    ev_stats = [];
    stats = {};

    modelPath = @(ep) fullfile(opts.expDir, sprintf('ac-epoch-%d.mat', ep));

    start = opts.continue * ac_findLastCheckpoint(opts.expDir) ;

    if start >= 1
        fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
        [ac, stats] = ac_loadState(modelPath(start)) ;
        for i = 1:start
            train_mse(i) = stats{i,1}.train.total_mse;
            test_mse(i) = stats{i,2}.val.total_mse;
        end
    end
    
f_mse = figure('Name', 'MSE');
f_val = figure('Name', 'VAL');
f_train = figure('Name', 'Train');
if opts.evaluate
    opts.epoch = start;
    [ac, ev_stats.train] = processEpoch_val(ac, ae, imdb, opts, 'train', mnist_x, f_val) ;
    [ac, ev_stats.val] = processEpoch_val(ac, ae, imdb, opts, 'val', mnist_x, f_val) ;
    return; 
end

for epoch=start+1:opts.numepochs
    
    params = opts ;
    params.epoch = epoch ;
    [ac, stats{end+1,1}.train] = processEpoch_train(ac, ae, imdb, params, f_train) ;
    [ac, stats{end,2}.val] = processEpoch_val(ac, ae, imdb, params, 'val', mnist_x, f_val) ;
    train_mse(epoch) = stats{epoch,1}.train.total_mse;
    test_mse(epoch) = stats{epoch,2}.val.total_mse;
    figure(f_mse);
    subplot(1,2,1);
    plot(train_mse); drawnow;
    subplot(1,2,2);
    plot(test_mse); drawnow;
    save(fullfile(opts.expDir, sprintf('ac-epoch-%d.mat', epoch)), 'ac', 'stats', '-v7.3');
end
ev_stats.train = train_mse;
ev_stats.val = test_mse;
end

%-------------------------------------------------------------------
function [ac, stats] = processEpoch_train(ac, ae, imdb, params, f_train)
%------------------------------------------------------------------   
subset = params.train;
start = tic;
acc_loss = 0;
acc_mse = 0;
num = 0;
stats.num = 0;
stats.time = 0;
adjustTime = 0;
stats.batch_loss = [];
stats.avg_loss = [];
stats.total_loss = 0;
stats.total_mse = 0;
stats.batch_mse = [];
stats.avg_mse = [];

ae_params = params;

for n=1:params.batch_size:numel(subset)
    fprintf('train: epoch %02d: %3d/%3d:', params.epoch, ...
        fix((n-1)/params.batch_size)+1, ceil(numel(subset)/params.batch_size)) ;
    index = fix((n-1)/params.batch_size)+1;
    batchStart = n;
    batchEnd = min(n+params.batch_size-1, numel(subset)) ;
    batch = subset(batchStart : 1 : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end
    [im, hs, coch, ~, label] = getBatch(imdb, batch) ;
    coch_spike = pixel_to_spike(coch, params.dt, params.duration, params.max_rate);
    ac = ac_initialize(ac, params);
    ae = ae.initialize(ae_params);
    output = zeros(size(im));
 
    for t = 1:size(params.dt:params.dt:params.duration,2)  
        ac = ac_ff(ac, coch_spike(:,:,t), params); 
        %fprintf('%f',mod(t, params.T)+1 );
        [ac, loss] = ac_calculate_loss(ac, hs(:,:,mod(t, params.T)+1), params);
        ac = ac_calcgrad(ac, params, t);        
        ae.hidden.spikes = ac.layers{2}.spikes;
        ae = ae.decode_test(ae_params);
        output = output + ae.output.spikes;
    end
    output = zscore(output);
    im = zscore(im);
    ae_mse = mse(im, output);     
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    acc_loss = acc_loss + loss;
    acc_mse = acc_mse + ae_mse;
    stats.batch_loss(index) = loss/numel(batch);
    stats.avg_loss(index) = acc_loss/num;
    stats.batch_mse(index) = ae_mse;
    stats.avg_mse(index) = acc_mse/index;
    stats.num(index) = num ;
    stats.time = time ;
    %currentSpeed = batchSize / batchTime ;
    %averageSpeed = (n + batchSize - 1) / time ;
    if n == 3*params.batch_size + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    fprintf(' loss: %.3f avgloss: %1.3f mse: %1.3f avgmse: %1.3f \n', ...
               stats.batch_loss(index), stats.avg_loss(index), stats.batch_mse(index), stats.avg_mse(index));
        

   
end
stats.total_loss = stats.avg_loss(end);
stats.total_mse = stats.avg_mse(end);

% pick a random spectrogram from training set
idx1 = 1;
l = label(idx1);
spike_input = coch_spike(:,idx1, :);
%feed it to the trained network
params2 = params;
params2.batch_size = 1;
ae = ae.initialize(params2);
ac = ac_initialize(ac, params2);
train_output_spikes = zeros(ae.output_size,1);
for t = 1:params2.duration/params2.dt
    ac = ac_ff_test(ac, spike_input(:,:,t), params2);
    ae.hidden.spikes = ac.layers{2}.spikes;
    ae = ae.decode_test(params2);
    train_output_spikes = train_output_spikes + ae.output.spikes;
    %debug_plot(train_output_spikes);
end


% pick a random image from testing set
l2 = imdb.label(params.val);
l3 = find(l2(:,1)==l);
idx2 = params.val(l3(1));
spike_input = pixel_to_spike(imdb.coch(:,idx2), params.dt, params.duration, params.max_rate);
ae = ae.initialize(params2);
ac = ac_initialize(ac, params2);
test_output_spikes = zeros(size(ae.output_size,1));

for t = 1:params2.duration/params2.dt
    ac = ac_ff_test(ac, spike_input(:,:,t), params2);
    ae.hidden.spikes = ac.layers{2}.spikes;
    ae = ae.decode_test(params2);
    test_output_spikes = test_output_spikes + ae.output.spikes;
    %debug_plot(test_output_spikes);
end
  
ac_fig_plot(f_train, im(:,idx1), train_output_spikes, imdb.image(:,idx2), test_output_spikes, stats.batch_loss, stats.batch_mse);

end


% -------------------------------------------------------------------------
function [ac, stats] = ac_loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'ac', 'stats') ;

if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end


end

% -------------------------------------------------------------------------
% function saveStats(fileName, stats)
% % -------------------------------------------------------------------------
% if exist(fileName)
%   save(fileName, 'stats', '-append') ;
% else
%   save(fileName, 'stats') ;
% end
% 
% end

%--------------------------------------------------------------
function [im, hs, coch, audio, label] = getBatch(imdb, batch)
%----------------------------------------------------------    
im = imdb.image(:,batch) ;
coch = imdb.coch(:,batch) ;
label = imdb.label(batch) ;
audio = imdb.audio(batch);
hs = imdb.hidden_state(:,batch, :);

end


%-------------------------------------------------------------------
function [ac, stats] = processEpoch_val(ac, ae, imdb, params, mode, mnist_x, f_val)
%------------------------------------------------------------------   
subset = params.(mode);
num = 0;
stats.total_mse = 0;
params.batch_size = 100;
ae_params = params;

for n=1:params.batch_size:numel(subset)
    fprintf('val: epoch %02d: %3d/%3d:', params.epoch, ...
        fix((n-1)/params.batch_size)+1, ceil(numel(subset)/params.batch_size)) ;
    index = fix((n-1)/params.batch_size)+1;
    %batchSize = min(params.batch_size, numel(subset) - n + 1) ;
    batchStart = n;
    batchEnd = min(n+params.batch_size-1, numel(subset)) ;
    batch = subset(batchStart : 1 : batchEnd) ;
    num = num + numel(batch) ;
    params.batch_size = min(100, numel(batch));
    ae_params.batch_size = params.batch_size;
    if numel(batch) == 0, continue ; end
    [im, ~, coch, ~, label] = getBatch(imdb, batch) ;
    coch_spike = pixel_to_spike(coch, params.dt, params.duration, params.max_rate);
    ac = ac_initialize(ac, params);
    ae = ae.initialize(ae_params);
    output = zeros(size(im));
 
    for t = 1:size(params.dt:params.dt:params.duration,2)  
        ac = ac_ff(ac, coch_spike(:,:,t), params);     
        ae.hidden.spikes = ac.layers{2}.spikes;
        ae = ae.decode_test(ae_params);
        output = output + ae.output.spikes;
    end
    output = zscore(output);
    im = zscore(im);
    ae_mse = zeros(1,size(output,2));
    %best_image = zeros(784. size(output,2));
    for k = 1:size(output,2)
       [ae_mse(1, k), ~] = min_mse(output(:,k), label(k), mnist_x);         
    end 
    ae_mse_all = sum(ae_mse, 2)/params.batch_size;
    %ae_mse = mse(im, output); 
    stats.total_mse = stats.total_mse + ae_mse_all;
    
    fprintf(' mse: %1.3f \n', ...
              stats.total_mse/index);  
    
    
end
stats.total_mse = stats.total_mse/index;
% pick an audio sample
idx = randi(size(coch_spike, 2));
spike_input = coch_spike(:,idx, :);
%feed it to the trained network
params2 = params;
params2.batch_size = 1;
ae = ae.initialize(params2);
ac = ac_initialize(ac, params2);
output_spikes = zeros(ae.output_size,1);
for t = 1:params2.duration/params2.dt
    ac = ac_ff_test(ac, spike_input(:,:,t), params2);
    ae.hidden.spikes = ac.layers{2}.spikes;
    ae = ae.decode_test(params2);
    output_spikes = output_spikes + ae.output.spikes;
end 
output_spikes = zscore(output_spikes);
[~, image] = min_mse(output_spikes,label(idx),mnist_x);
  
ac_fig_plot_val(f_val, im(:,idx), output_spikes, image);
%ac_fig_plot_val_images_v2(f_val, im(:,idx), output_spikes);
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

% END FUNCTION

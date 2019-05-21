% -------------------------------------------------------------------------
function epoch = ac_findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
        list = dir(fullfile(modelDir, 'ac-epoch-*.mat')) ;
        tokens = regexp({list.name}, 'ac-epoch-([\d]+).mat', 'tokens') ;
        epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
        epoch = max([epoch 0]) ;
end

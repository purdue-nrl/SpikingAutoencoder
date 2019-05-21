function [ae, train_loss, test_loss, train_mse, test_mse] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'ae', 'train_loss', 'test_loss', 'train_mse', 'test_mse') ;

end

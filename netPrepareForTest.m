% Prepare a network for 'test' mode
% - disable `precious`
% - optionally remove `dropout`

function net= netPrepareForTest(net, removeDropout)
    if nargin<2, removeDropout= false; end
    
    for iLayer= 1:length(net.layers)
        net.layers{iLayer}.precious= false;
    end
    
    if removeDropout
        net.layers( ismember( relja_layerTypes(net), {'dropout'} ) )= [];
    end
    
end

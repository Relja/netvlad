% Prepare a network for 'test' mode
% - disable `precious`
% - optionally remove `dropout`

function net= netPrepareForTest(net, removeDropout, removeExtras)
    if nargin<2, removeDropout= false; end
    if nargin<3, removeExtras= false; end
    
    for iLayer= 1:length(net.layers)
        net.layers{iLayer}.precious= false;
        
        if removeExtras
            if isFieldOrProp(net.layers{iLayer}, 'momentum')
                net.layers{iLayer}.momentum= [];
            end
        end
    end
    
    if removeDropout
        net.layers( ismember( relja_layerTypes(net), {'dropout'} ) )= [];
    end
    
end



function is= isFieldOrProp(l, propName)
    is= isprop(l, propName) || isfield(l, propName);
end

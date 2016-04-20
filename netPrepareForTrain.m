function net= netPrepareForTrain(net, backPropToLayer)
    if nargin<2, backPropToLayer= 1; end
    
    nLayers= numel(net.layers);
    for i= 1:nLayers
        if i>=backPropToLayer && isFieldOrProp(net.layers{i}, 'weights')
            J= numel(net.layers{i}.weights);
            for j= 1:J
                net.layers{i}.momentum{j}= zeros(size(net.layers{i}.weights{j}), 'single');
            end
            if shouldAdd(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate= ones(1, J, 'single');
            end
            if shouldAdd(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay= ones(1, J, 'single');
            end
        end
        
        % --- This is a bit of a hack:
        % When doing the forward pass during training, we have to keep all
        % intermediate values in order to do gradient computations via
        % backprop. However, ReLU inputs can be forgotten as is already done
        % in MatConvNet when the backprop is done at the same time as the
        % forward pass, and this provides quite a lot of memory saving for
        % some networks (e.g. VGG-16). In order to achieve the desired behaviour
        % (remember everything apart from ReLU input), we will actually use
        % conserveMemory=true (i.e. forget everything), but then we explicitly
        % set precious= true for every layer apart from the one before ReLU.
        % We can further save memory by forgetting all values below backPropToLayer,
        % as they are not needed if you are doing only partial backprop
        % (this can also be done automatically with vl_simplenn but only if
        % the backward pass is done simultaneously with the forward).
        % So in the end, we mark precious only layers >=backPropToLayer-1 & !before-ReLU
        if i>=backPropToLayer-1 && (i<nLayers && ~strcmp(net.layers{i+1}.type, 'relu'))
            net.layers{i}.precious= true;
        else
            net.layers{i}.precious= false;
        end
    end
end



function is= isFieldOrProp(l, propName)
    is= isprop(l, propName) || isfield(l, propName);
end



function should= shouldAdd(l, propName)
    if ~isa(l, 'struct')
        assert(isprop(l, propName));
        should= isempty(l.(propName));
    else
        should= ~isfield(l, propName) || isempty(l.(propName));
    end
end

function net= netPrepareForTrain(net)
    for i= 1:numel(net.layers)
        if isFieldOrProp(net.layers{i}, 'weights')
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
    end
end



function is= isFieldOrProp(l, propName)
    if strcmp(l.type, 'custom')
        is= isprop(l, propName);
    else
        is= isfield(l, propName);
    end
end



function should= shouldAdd(l, propName)
    if strcmp(l.type, 'custom')
        assert(isprop(l, propName));
        should= isempty(l.(propName));
    else
        should= ~isfield(l, propName) || isempty(l.(propName));
    end
end

function cleanup(sessionID, doDelete)
    % Deletes intermediate sessionID_ep*_latest.mat files created during
    % training, apart from the one corresponding to the best network.
    % Careful - if your criterion for picking the best network is different
    % you might delete too much.
    % For an additional precaution, `doDelete` should be specified and `true`
    % for the actual deletion to happen, otherwise just prints what would
    % be deleted if `doDelete` was true.
    if nargin<2, doDelete= false; end
    
    paths= localPaths();
    load( sprintf('%s%s_latest.mat', paths.outPrefix, sessionID), 'obj' );
    nEpochs= size(obj.val.recall, 2);
    
    bestEpoch= pickBestNet(sessionID, 5, false);
    
    for iEpoch= 1:(nEpochs+1)
        if iEpoch==bestEpoch, continue; end
        fn= sprintf('%s%s_ep%06d_latest.mat', paths.outPrefix, sessionID, iEpoch);
        if exist(fn, 'file')
            if doDelete
                relja_display('Deleting: %s', fn);
                delete(fn);
            else
                relja_display('Would delete: %s', fn);
            end
        end
    end
end

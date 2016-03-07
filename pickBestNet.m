function [bestEpoch, bestNet]= pickBestNet(sessionID, N, verbose)
    if nargin<2, N= 5; end
    if nargin<3, verbose= 2; end
    
    paths= localPaths();
    outFnLatest= sprintf('%s%s_latest.mat', paths.outPrefix, sessionID);
    res= load( outFnLatest, 'obj', 'opts', 'auxData');
    
    if res.opts.epochTestFrequency~=1
        error('This code assumes epochTestFrequency==1 (it is %d)', res.opts.epochTestFrequency);
    end
    bestEpoch= getBestEpoch(res.obj.val.recall, res.opts.recallNs, N);
    assert(~isempty(bestEpoch));
    
    if verbose>0
        whichRecallInds= 1:6;
        hline= repmat('=', 1, length(whichRecallInds)*5-1 + 14);
        
        relja_display('%s Best epoch: %d (out of %d)', sessionID, bestEpoch, size(res.obj.val.recall,2) );
        
        if isfield(res.obj, 'pretrain')
            offtheshelfValRecs= res.obj.pretrain.val.recall( whichRecallInds );
        end
        bestValRecs= res.obj.val.recall( whichRecallInds, bestEpoch );
        
        relja_display('%s', hline);
        recallStr= sprintf('%04d ', res.opts.recallNs(whichRecallInds) );
        relja_display('Recall@N      %s', recallStr);
        relja_display('%s', hline);
        
        if exist('offtheshelfValRecs', 'var')
            offtheshelfStr= sprintf('%.2f ', offtheshelfValRecs);
        end
        trainedStr= sprintf('%.2f ', bestValRecs);
        
        relja_display('off-the-shelf %s', offtheshelfStr);
        relja_display('our trained   %s', trainedStr);
        
        if exist('offtheshelfValRecs', 'var')
            relImpStr= sprintf('%.2f ', bestValRecs./offtheshelfValRecs);
            relja_display('trained/shelf %s', relImpStr);
        end
        
        if verbose>1
            
            figure; plotResults(res.obj, res.opts, res.auxData);
            
            figure;
            plot( res.opts.recallNs, res.obj.pretrain.val.recall, 'bx-' );
            hold on;
            plot( res.opts.recallNs, res.obj.val.recall(:, bestEpoch), 'ro-' );
            xlabel('N');
            ylabel('Recall@N');
            xlim([0,50]);
            grid on;
            title( sprintf('%s %s %s %s %s', sessionID, res.opts.netID, res.opts.layerName, res.opts.dbValName, res.opts.method), 'Interpreter', 'none' );
            legend( 'off-the-shelf', 'our trained', 'Location', 'SouthEast');
        end
    end
    
    if nargout>1
        outFnBest= sprintf('%s%s_ep%06d_latest.mat', paths.outPrefix, sessionID, bestEpoch);
        load(outFnBest, 'net');
        bestNet= net; clear net;
        bestNet.meta.sessionID= sessionID;
        bestNet.meta.epoch= bestEpoch;
        
        % remove unneeded momentum
        for iLayer= 1:length(bestNet.layers)
            if isfield(bestNet.layers{iLayer}, 'momentum')
                bestNet.layers{iLayer}= rmfield(bestNet.layers{iLayer}, 'momentum');
            elseif isprop(bestNet.layers{iLayer}, 'momentum')
                bestNet.layers{iLayer}.momentum= [];
            end
        end
    end
end

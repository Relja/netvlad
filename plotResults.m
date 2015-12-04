function plotResults(obj, opts, auxData, startVals, w)
    if nargin<3, auxData= []; end
    if nargin<4, startVals= []; end
    if nargin<5
        w= min( ceil( 0.05 * length(obj.train.loss) ), 5000 );
    end
    if ~isempty(auxData) && isfield(auxData, 'numTrain')
        numTrain= auxData.numTrain;
    elseif isfield(opts, 'numTrain')
        % for back-compatibility
        numTrain= opts.numTrain;
    else
        numTrain= 1; % false
    end
    if isfield(obj, 'pretrain')
        if ~isempty(startVals)
            warning('startVals is given while obj.pretrain exists, using obj.pretrain');
        end
        startVals= obj.pretrain;
    end
    
    clf;
    set(gcf, 'Color', 'w');
    
    if ~isempty(obj.train.rankloss)
        subplot(2,2,1);
    end
    
    if length(obj.train.loss)>w
        s= smooth(obj.train.loss, w); s(1:min(round(w/2),end))= NaN; s(max(1, end-round(w/2)):end)= NaN;
        semilogy( (1:length(s))*opts.batchSize/numTrain, s, 'b-');
        title( 'train dynamic loss (smoothed)' );
    else
        semilogy(obj.train.loss, 'r-');
        title('train dynamic loss');
    end
    grid on;
    xlabel( 'epoch' );
    axis tight;
    
    if isempty(obj.train.rankloss), drawnow; return; end
    
    tRec= obj.train.recall;
    vRec= obj.val.recall;
    tRank= obj.train.rankloss;
    vRank= obj.val.rankloss;
    
    epochs= [1:size(obj.train.rankloss,2)]*opts.epochTestFrequency;
    
    if ~isempty(startVals)
        assert(size(startVals.train.recall,1)==length(opts.recallNs));
        epochs= [0, epochs];
        tRec= [startVals.train.recall(:,end), tRec];
        vRec= [startVals.val.recall(:,end), vRec];
        tRank= [startVals.train.rankloss(:,end), tRank];
        vRank= [startVals.val.rankloss(:,end), vRank];
    end
    
    subplot(2,2,3);
    semilogy( epochs, tRank, 'rx--');
    hold on;
    semilogy( epochs, vRank, 'rx-');
    legend('train.Q', 'val.Q', 'Location', 'North', 'Orientation', 'horizontal');
    grid on;
    title('loss');
    xlabel( 'epoch' );
    axis tight;
    
    if isempty(obj.train.recall), drawnow; return; end
    
    subplot(1,2,2); hold on; grid on;
    cols= {'r', 'g', 'b', 'm', 'c'};
    whichRecalls= [1,2,5,10,30];
    whichToPlot= ismember(opts.recallNs, whichRecalls); assert(any(whichToPlot));
    inds= find(whichToPlot);
    assert(length(cols)>=length(inds));
    
    for i= 1:length(inds)
        plot(epochs, 100*tRec( inds(i), :), [cols{i}, 'o--']);
        plot(epochs, 100*vRec( inds(i), :), [cols{i}, 'x-']);
    end
    hold off;
    
    legendEntries= {};
    for i= 1:length(whichRecalls)
        r= whichRecalls(i);
        legendEntries=[legendEntries, sprintf('t.%d', r), sprintf('v.%d', r)];
    end
    
    axis tight;
    title('recall');
    xlabel('epoch');
    axis tight;
    legend(legendEntries{:}, 'Location', 'NorthWestOutside');
    
    drawnow;
end



function z= longest(x,y)
    if size(x,2)>size(y,2), z=x; else, z=y; end
end

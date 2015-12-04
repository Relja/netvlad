function [rankloss, losses]= testCoreRank(db, qFeat, dbFeat, margin, nNegChoice, varargin)
    opts= struct(...
        'nTestSample', 1000, ...
        'recallNs', [1:5, 10:5:100], ...
        'printN', 10 ...
        );
    opts= vl_argparse(opts, varargin);
    assert(~isinf(opts.nTestSample));
    
    rngState= rng;
    
    nQueries= size(qFeat, 2);
    if opts.nTestSample < nQueries
        rng(43);
        toTest= randsample(nQueries, opts.nTestSample);
    else
        toTest= 1:nQueries;
        opts.nTestSample= nQueries;
    end
    
    losses= [];
    
    rng(43);
    evalProg= tic;
    
    for iTestSample= 1:opts.nTestSample
        
        relja_progress(iTestSample, ...
                       opts.nTestSample, ...
                       sprintf('%.4f', mean(losses)), evalProg);
        
        qID= toTest(iTestSample);
        potPosIDs= db.nontrivialPosQ(qID);
        if isempty(potPosIDs), continue; end
        negIDs= db.sampleNegsQ(qID, nNegChoice);
        
        dsSq= sum( bsxfun(@minus, qFeat(:, qID), dbFeat(:, potPosIDs)) .^2, 1 );
        dPos= min(dsSq);
        dsSq= sum( bsxfun(@minus, qFeat(:, qID), dbFeat(:, negIDs)) .^2, 1 );
        losses(end+1)= mean( max(dPos + margin - dsSq, 0) );
    end
    t= toc(evalProg);
    
    if isempty(losses)
        error('No positives in the test');
    end
    rankloss= mean(losses);
    relja_display('\n\tloss= %.4f, margin= %.4f, time= %.4f s, avgTime= %.4f ms\n', rankloss, margin, t, t*1000/length(toTest));
    
    rng(rngState);
end

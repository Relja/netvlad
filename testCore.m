function [recalls, allRecalls]= testCore(db, qFeat, dbFeat, varargin)
    opts= struct(...
        'nTestSample', inf, ...
        'recallNs', [1:5, 10:5:100], ...
        'printN', 10 ...
        );
    opts= vl_argparse(opts, varargin);
    
    searcherRAW_= @(iQuery, nTop) rawNnSearch(qFeat(:,iQuery), dbFeat, nTop);
    if ismethod(db, 'nnSearchPostprocess')
        searcherRAW= @(iQuery, nTop) db.nnSearchPostprocess(searcherRAW_, iQuery, nTop);
    else
        searcherRAW= searcherRAW_;
    end
    [res, recalls]= recallAtN( searcherRAW, db.numQueries,  @(iQuery, iDb) db.isPosQ(iQuery, iDb), opts.recallNs, opts.printN, opts.nTestSample );
    
    allRecalls= recalls;
    recalls= mean( allRecalls, 1 )';
    
end

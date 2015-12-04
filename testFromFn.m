function [recall, rankloss, allRecalls, opts]= testFromFn(db, dbFeatFn, qFeatFn, opts, varargin)
    if nargin<4
        opts= struct(...
            'nTestRankSample', 0, ...
            'nTestSample', inf, ...
            'recallNs', [1:5, 10:5:100], ...
            'margin', 0.1, ...
            'nNegChoice', 1000 ...
            );
    end
    
    relja_display('testFromFn:\n%s\n%s', dbFeatFn, qFeatFn);
    
    qFeat= fread( fopen(qFeatFn, 'rb'), inf, 'float32=>single');
    qFeat= reshape(qFeat, [], db.numQueries);
    nDims= size(qFeat, 1);
    dbFeat= fread( fopen(dbFeatFn, 'rb'), [nDims, db.numImages], 'float32=>single');
    assert(size(dbFeat,2)==db.numImages);
    
    if opts.nTestRankSample>0
        rankloss= testCoreRank(db, qFeat, dbFeat, opts.margin, opts.nNegChoice, 'nTestSample', opts.nTestRankSample);
    else
        rankloss= [];
    end
    [recall, allRecalls]= testCore(db, qFeat, dbFeat, 'nTestSample', opts.nTestSample, 'recallNs', opts.recallNs);
end

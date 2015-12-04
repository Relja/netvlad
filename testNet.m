function [recall, rankloss]= testNet(db, net, opts, ID, qFeat, dbFeat)
    
    relja_display('testNet: %s %s', opts.sessionID, ID);
    
    rankloss= testCoreRank(db, qFeat, dbFeat, opts.margin, opts.nNegChoice, 'nTestSample', opts.nTestRankSample);
    recall= testCore(db, qFeat, dbFeat, 'nTestSample', opts.nTestSample, 'recallNs', opts.recallNs);
end

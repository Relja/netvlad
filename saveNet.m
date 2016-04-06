function saveNet(net, obj, opts, auxData, ID, description, verbose);
    if nargin<7, verbose= false; end
    if verbose
        relja_display('saving net %s %s : %s', opts.sessionID, ID, description);
    end
    if isempty(auxData)
        auxData= {};
    end
    auxData.rngState= rng;
    
    outFnCurrent= sprintf('%s%s_%s.mat', opts.outPrefix, opts.sessionID, ID);
    outFnLatest= sprintf('%s%s_%s.mat', opts.outPrefix, opts.sessionID, 'latest');
    save( outFnCurrent, 'net', 'obj', 'opts', 'auxData', '-v7.3');
    
    net= netPrepareForTest(net);
    relja_netSave(net, outFnCurrent);
    
    copyfile(outFnCurrent, outFnLatest);
end

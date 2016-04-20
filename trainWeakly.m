function sessionID= trainWeakly(dbTrain, dbVal, varargin)
    
    % See README_more.md for explanations
    
    opts= struct(...
        'netID', 'caffe', ...
        'layerName', 'conv5', ...
        'method', 'vlad_preL2_intra', ...
        'batchSize', 4, ...
        'learningRate', 0.0001, ...
        'lrDownFreq', 5, ...
        'lrDownFactor', 2, ...
        'weightDecay', 0.001, ...
        'momentum', 0.9, ...
        'backPropToLayer', 1, ...
        'fixLayers', [], ...
        'nNegChoice', 1000, ...
        'nNegCap', 10, ...
        'nNegCache', 10, ...
        'nEpoch', 30, ...
        'margin', 0.1, ...
        'excludeVeryHard', false, ...
        'jitterFlip', false, ...
        'jitterScale', [], ...
        'sessionID', [], ...
        'outPrefix', [], ...
        'dbCheckpoint0', [], ...
        'qCheckpoint0', [], ...
        'dbCheckpoint0val', [], ...
        'qCheckpoint0val', [], ...
        'checkpoint0suffix', '', ...
        'info', '', ...
        'test0', true, ...
        'saveFrequency', 2000, ...
        'compFeatsFrequency', 1000, ...
        'computeBatchSize', 10, ...
        'epochTestFrequency', 1, ... % recommended not to be changed (pickBestNet won't work otherwise)
        'doDraw', false, ...
        'printLoss', false, ...
        'printBatchLoss', false, ...
        'nTestSample', 1000, ...
        'nTestRankSample', 5000, ...
        'recallNs', [1:5, 10:5:100], ...
        'useGPU', true, ...
        'numThreads', 12, ...
        'startEpoch', 1 ...
        );
    paths= localPaths();
    
    opts= vl_argparse(opts, varargin);
    if isempty(opts.sessionID),
        if opts.startEpoch>1, error('Have to specify sessionID to restart'); end
        rng('shuffle'); opts.sessionID= relja_randomHex(4);
    end
    sessionID= opts.sessionID;
    if isempty(opts.outPrefix)
        opts.outPrefix= paths.outPrefix;
    end
    opts.dbTrainName= dbTrain.name;
    opts.dbValName= dbVal.name;
    if isempty(opts.fixLayers), opts.fixLayers= {}; end;
    if ~isempty(opts.jitterScale)
        im= imread([dbTrain.dbPath, dbTrain.dbImageFns{1}]);
        origImS= min(size(im,1), size(im,2));
    end
    
    
    
    
    if opts.startEpoch<2
        
        % ----- Checkpoint names
        
        if ~isempty(opts.checkpoint0suffix)
            opts.checkpoint0suffix= [opts.checkpoint0suffix, '_'];
        end
        if isempty(opts.dbCheckpoint0)
            opts.dbCheckpoint0= sprintf('%s%s_%s_%s_%s_%sdb.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.qCheckpoint0)
            opts.qCheckpoint0= sprintf('%s%s_%s_%s_%s_%sq.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.dbCheckpoint0val)
            opts.dbCheckpoint0val= sprintf('%s%s_%s_%s_%s_%sdb.bin', opts.outPrefix, dbVal.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        if isempty(opts.qCheckpoint0val)
            opts.qCheckpoint0val= sprintf('%s%s_%s_%s_%s_%sq.bin', opts.outPrefix, dbVal.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix);
        end
        
        % ----- Network setup
        
        net= loadNet(opts.netID, opts.layerName);
        
        % --- Add my layers
        net= addLayers(net, opts, dbTrain);
        
        % --- BackProp depth
        if isempty(opts.backPropToLayer)
            opts.backPropToLayer= 1;
        else
            if ~isnumeric( opts.backPropToLayer )
                assert( isstr(opts.backPropToLayer) );
                opts.backPropToLayer= relja_whichLayer(net, opts.backPropToLayer);
            end
        end
        opts.backPropToLayerName= net.layers{opts.backPropToLayer}.name;
        opts.backPropDepth= length(net.layers)-opts.backPropToLayer+1;
        assert( all(ismember(opts.fixLayers, relja_layerNames(net))) );
        
        display(opts);
        
        
        
        % ----- Init
        
        auxData= {};
        auxData.epochStartTime= {};
        auxData.numTrain= dbTrain.numQueries;
        auxData.negCache= cell(dbTrain.numQueries, 1);
        
        obj= struct();
        obj.train= struct('loss', [], 'recall', [], 'rankloss', []);
        obj.val= struct('loss', [], 'recall', [], 'rankloss', []);
        
    else
        
        % ----- Continue from an epoch
        
        ID= sprintf('ep%06d_latest', opts.startEpoch-1);
        outFnCurrent= sprintf('%s%s_%s.mat', opts.outPrefix, opts.sessionID, ID);
        tmpopts= opts;
        load(outFnCurrent, 'net', 'obj', 'opts', 'auxData'); % rewrites opts
        clear ID outFnCurrent;
        
        opts.startEpoch= tmpopts.startEpoch;
        opts.test0= false;
        opts.useGPU= tmpopts.useGPU;
        opts.numThreads= tmpopts.numThreads;
        
        if ~isfield(opts, 'dbCheckpoint0_orig')
            opts.dbCheckpoint0_orig= opts.dbCheckpoint0;
            opts.qCheckpoint0_orig= opts.qCheckpoint0;
        end
        opts.dbCheckpoint0= tmpopts.dbCheckpoint0;
        opts.qCheckpoint0= tmpopts.qCheckpoint0;
        
        if isempty(opts.qCheckpoint0)
            opts.dbCheckpoint0= sprintf('%s%s_%s_%s_%s_%s%s_ep%06d_db.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix, opts.sessionID, opts.startEpoch-1);
        end
        if isempty(opts.qCheckpoint0)
            opts.qCheckpoint0= sprintf('%s%s_%s_%s_%s_%s%s_ep%06d_q.bin', opts.outPrefix, dbTrain.name, opts.netID, opts.layerName, opts.method, opts.checkpoint0suffix, opts.sessionID, opts.startEpoch-1);
        end
        
        display(opts);
        
    end
    
    % --- Prepare for train
    net= netPrepareForTrain(net, opts.backPropToLayer);
    
    if opts.useGPU
        net= relja_simplenn_move(net, 'gpu');
    end
    
    nBatches= floor( dbTrain.numQueries / opts.batchSize ); % some might be cut, no biggie
    batchSaveFrequency= ceil(opts.saveFrequency/opts.batchSize);
    batchCompFeatsFrequency= ceil(opts.compFeatsFrequency/opts.batchSize);
    
    
    
    % ----- Initial features
    
    if ~exist(opts.qCheckpoint0, 'file')
        serialAllFeats(net, dbTrain.qPath, dbTrain.qImageFns, ...
            opts.qCheckpoint0, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    end
    if ~exist(opts.dbCheckpoint0, 'file')
        serialAllFeats(net, dbTrain.dbPath, dbTrain.dbImageFns, ...
            opts.dbCheckpoint0, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    end
    if opts.test0
        if ~exist(opts.qCheckpoint0val, 'file')
            serialAllFeats(net, dbVal.qPath, dbVal.qImageFns, ...
                opts.qCheckpoint0val, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
        end
        if ~exist(opts.dbCheckpoint0val, 'file')
            serialAllFeats(net, dbVal.dbPath, dbVal.dbImageFns, ...
                opts.dbCheckpoint0val, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
        end
        [obj.pretrain.val.recall, obj.pretrain.val.rankloss]= ...
            testFromFn(dbVal, opts.dbCheckpoint0val, opts.qCheckpoint0val, opts);
        [obj.pretrain.train.recall, obj.pretrain.train.rankloss]= ...
            testFromFn(dbTrain, opts.dbCheckpoint0, opts.qCheckpoint0, opts);
    end
    
    qFeat= fread( fopen(opts.qCheckpoint0, 'rb'), inf, 'float32=>single');
    qFeat= reshape(qFeat, [], dbTrain.numQueries);
    nDims= size(qFeat, 1);
    dbFeat= fread( fopen(opts.dbCheckpoint0, 'rb'), [nDims, dbTrain.numImages], 'float32=>single');
    
    
    assert( relja_netOutputDim(net)==nDims );
    
    
    
    % ----- Training
    
    lr= opts.learningRate;
    
    progEpoch= tic;
    
    for iEpoch= 1:opts.nEpoch
        relja_progress(iEpoch, opts.nEpoch, 'epoch', progEpoch);
        auxData.epochStartTime{end+1}= datestr(now);
        
        if iEpoch~=1 && rem(iEpoch, opts.lrDownFreq)==1
            oldLr= lr;
            lr= lr/opts.lrDownFactor;
            relja_display('Changing learning rate from %f to %f', oldLr, lr); clear oldLr;
            batchCompFeatsFrequency= round(batchCompFeatsFrequency*opts.lrDownFactor);
        end
        relja_display('Learning rate %f', lr);
        
        if opts.startEpoch>iEpoch, continue; end
        
        rng(43-1+iEpoch);
        trainOrder= randperm(dbTrain.numQueries);
        
        ID= sprintf('ep%06d_latest', iEpoch);
        trainID= sprintf('%s_train', ID);
        valID= sprintf('%s_val', ID);
        
        progBatch= tic;
        
        iQAbs= 1;
        iPosAbs= 2;
        
        for iBatch= 1:nBatches
            
            relja_progress(iBatch, nBatches, ...
                sprintf('%s epoch %d batch', opts.sessionID, iEpoch), progBatch);
            
            if rem(iBatch, batchSaveFrequency)==0
                saveNet(net, obj, opts, auxData, ID, sprintf('epoch %d batch %d', iEpoch, iBatch));
                if opts.doDraw, plotResults(obj, opts, auxData); end
            end
            
            if rem(iBatch, batchCompFeatsFrequency)==0 && iBatch~=1 && iBatch~=nBatches
                clear qFeat dbFeat;
                [qFeat, dbFeat]= computeAllFeats(dbTrain, net, opts, trainID, true);
            end
            
            
            
            qIDs= trainOrder( (iBatch-1)*opts.batchSize + (1:opts.batchSize) );
            
            losses= [];
            
            allRes= [];
            numTriplets= [];
            
            for iQuery= 1:opts.batchSize
                
                % ---------- compute closest positive and violating negatives
                
                qID= qIDs(iQuery);
                potPosIDs= dbTrain.nontrivialPosQ(qID);
                if isempty(potPosIDs), continue; end
                
                % ----- closest positive
                
                [posID, dPos]= yael_nn( ...
                    dbFeat(:, potPosIDs), ...
                    qFeat(:, qID), ...
                    1 );
                posID= potPosIDs(posID);
                
                % ----- hard negatives
                
                negIDs= unique([ ...
                    auxData.negCache{qID}; ...
                    dbTrain.sampleNegsQ(qID, opts.nNegChoice)]);
                
                % dsSq= sum( bsxfun(@minus, qFeat(:, qID), dbFeat(:, negIDs)) .^2, 1 )';
                [inds, dsSq]= yael_nn( ...
                    dbFeat(:, negIDs), ...
                    qFeat(:, qID), ...
                    min(opts.nNegCap*10, length(negIDs)) ... % 10x is hacky but fine
                    );
                negIDs= negIDs(inds);
                auxData.negCache{qID}= negIDs(1:min(opts.nNegCache, end));
                
                veryHardNegs= dsSq < dPos;
                violatingNegs= dsSq < dPos + opts.margin;
                if opts.excludeVeryHard
                    violatingNegs= violatingNegs & ~veryHardNegs;
                end
                nViolatingNegs= sum(violatingNegs);
                nVeryHardNegs= sum(veryHardNegs);
                if opts.printLoss
                    loss= sum( max(dPos + opts.margin - dsSq, 0) );
                    relja_display('%s loss= %.4f, #violate= %d, #vhard= %d, (qID=%d)', ...
                        opts.sessionID, loss, nViolatingNegs, nVeryHardNegs, qID);
                end
                
                if nViolatingNegs==0, losses(end+1)= 0; continue; end
                negIDs= negIDs(violatingNegs);
                dsSq= dsSq(violatingNegs);
                if nViolatingNegs>opts.nNegCap
                    [~, sortNegInds]= sort(dsSq); % not needed if using yael_nn below sampleNegsQ
                    negIDs= negIDs( sortNegInds(1:opts.nNegCap) );
                end
                
                % ---------- load images, normalize them
                
                imageFns= [ [dbTrain.qPath, dbTrain.qImageFns{qID}]; ...
                    strcat( dbTrain.dbPath, dbTrain.dbImageFns([posID; negIDs]) ) ];
                thisNumIms= length(imageFns);
                
                if isempty(opts.jitterScale)
                    ims_= vl_imreadjpeg(imageFns, 'numThreads', opts.numThreads);
                else
                    sc= opts.jitterScale( randsample(length(opts.jitterScale), 1) );
                    ims_= vl_imreadjpeg(imageFns, 'numThreads', opts.numThreads, 'Resize', round(sc*origImS));
                end
                
                % fix non-colour images
                for iIm= 1:thisNumIms
                    if size(ims_{iIm},3)==1
                        ims_{iIm}= cat(3,ims_{iIm},ims_{iIm},ims_{iIm});
                    end
                end
                ims= cat(4, ims_{:});
                
                ims(:,:,1,:)= ims(:,:,1,:) - net.meta.normalization.averageImage(1,1,1);
                ims(:,:,2,:)= ims(:,:,2,:) - net.meta.normalization.averageImage(1,1,2);
                ims(:,:,3,:)= ims(:,:,3,:) - net.meta.normalization.averageImage(1,1,3);
                
                if opts.jitterFlip && rand()>0.5
                    ims= ims(:,end:-1:1,:,:);
                end
                
                if opts.useGPU
                    ims= gpuArray(ims);
                end
                
                
                
                % ---------- forward
                
                res= vl_simplenn(net, ims, [], [], 'mode', 'normal', 'conserveMemory', true); % the memory saving related to backPropDepth is obayed implicitly due to running netPrepareForTrain before, see the comments in the function for an explanation
                if opts.backPropToLayer==1, res(1).x= ims; end % because of the 'conserveMemory' the input is deleted, restore it if needed
                ims= [];
                feats= reshape( gather(res(end).x), [], thisNumIms );
                
                % ---------- compute real distances and violating negs
                
                dsSq= sum( bsxfun(@minus, feats(:, 1), feats(:, 2:end)) .^2, 1 )';
                
                dPos= dsSq(1);
                veryHardNegs= dsSq(2:end) < dPos;
                violatingNegs= dsSq(2:end) < dPos + opts.margin;
                if opts.excludeVeryHard
                    violatingNegs= violatingNegs & ~veryHardNegs;
                end
                nViolatingNegs= sum(violatingNegs);
                nVeryHardNegs= sum(veryHardNegs);
                loss= sum( max(dPos + opts.margin - dsSq(2:end), 0) );
                if opts.printLoss
                    relja_display('   real loss= %.4f, #violate= %d, #vhard= %d, (qID=%d)', ...
                        loss, nViolatingNegs, nVeryHardNegs, qID);
                end
                losses(end+1)= loss;
                
                if nViolatingNegs==0, continue; end
                violatingNegAbsInds= 2+find(violatingNegs);
                
                
                % ---------- gradients
                
                dzdy= zeros(size(feats,1), thisNumIms, 'single');
                
                % 1-qID, 2-pos, 3:end-neg
                
                % grad(feat_query)
                dzdy(:, iQAbs)= 2*( ...
                    sum(feats(:, violatingNegAbsInds), 2) ...
                    - nViolatingNegs*feats(:, iPosAbs) );
                
                % grad(feat_pos)
                dzdy(:, iPosAbs)= 2* ...
                    nViolatingNegs * ( feats(:, iPosAbs)-feats(:, iQAbs) );
                
                % grad(feat_neg)
                dzdy(:, violatingNegAbsInds)= 2* (...
                     bsxfun(@minus, feats(:, iQAbs), feats(:, violatingNegAbsInds))  );
                
                if opts.useGPU
                    dzdy= gpuArray(dzdy);
                end
                
                
                
                % ---------- backward pass
                allRes= [allRes; ...
                    vl_simplenn(net, ims, dzdy, res, ...
                        'mode', 'normal', ...
                        'skipForward', true, ...
                        'backPropDepth', opts.backPropDepth, ...
                        'conserveMemory', true)];
                numTriplets= [numTriplets, nViolatingNegs];
                
            end % for sample in batch
            
            clear res;
            
            if isempty(losses)
                loss= 0;
            else
                loss= mean(losses);
            end
            
            obj.train.loss(end+1)= loss;
            if opts.printBatchLoss
                relja_display('%s batchloss= %.4f', opts.sessionID, loss);
            end
            
            
            
            thisBatchSize= sum(numTriplets);
            
            if thisBatchSize > 0
                
                % ---------- train
                
                for l= 1:numel(net.layers)
                    for j= 1:numel(allRes(1, l).dzdw)
                        if ismember(net.layers{l}.name, opts.fixLayers) continue; end
                        
                        dzdw= allRes(1, l).dzdw{j};
                        for iQuery= 2:size(allRes,1)
                            dzdw= dzdw + allRes(iQuery, l).dzdw{j};
                        end
                        
                        thisDecay= opts.weightDecay * net.layers{l}.weightDecay(j);
                        thisLR= lr * net.layers{l}.learningRate(j);
                        
                        net.layers{l}.momentum{j}= ...
                            opts.momentum * net.layers{l}.momentum{j} ...
                            - thisDecay * net.layers{l}.weights{j} ...
                            - (1 / thisBatchSize) * dzdw;
                        net.layers{l}.weights{j}= net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j};
                    end
                end
                
                clear dzdw;
            end
            
            clear allRes;
            
        end % for batch
        
        
        
        clear qFeat dbFeat;
        ID= sprintf('ep%06d_latest', iEpoch);
        saveNet(net, obj, opts, auxData, ID, sprintf('epoch-end %d', iEpoch));
        
        testNow= iEpoch==opts.nEpoch || rem(iEpoch, opts.epochTestFrequency)==0;
        
        if testNow
            [qFeatVal, dbFeatVal]= computeAllFeats(dbVal, net, opts, valID, true);
            [obj.val.recall(:, end+1), obj.val.rankloss(:, end+1) ...
                ]= testNet(dbVal, net, opts, valID, qFeatVal, dbFeatVal);
            clear qFeatVal dbFeatVal;
        end
        
        [qFeat, dbFeat]= computeAllFeats(dbTrain, net, opts, trainID, true);
        
        if testNow
            [obj.train.recall(:, end+1), obj.train.rankloss(:, end+1) ...
                ]= testNet(dbTrain, net, opts, trainID, qFeat, dbFeat);
            
            % to save the results
            saveNet(net, obj, opts, auxData, ID, sprintf('epoch-end %d', iEpoch));
            
            if opts.doDraw, plotResults(obj, opts, auxData); end
        end
        
    end % for epoch
    
end



function [qFeat, dbFeat]= computeAllFeats(db, net, opts, ID, delFile)
    if nargin<5, delFile= true; end
    outPrefix= sprintf('%s%s_%s', opts.outPrefix, opts.sessionID, ID);
    
    qFeatFn= sprintf('%s_q.bin', outPrefix);
    tmpFn= sprintf('%s.tmp', qFeatFn);
    serialAllFeats(net, db.qPath, db.qImageFns, ...
        tmpFn, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    movefile(tmpFn, qFeatFn);
    
    qFeat= fread( fopen(qFeatFn, 'rb'), inf, 'float32=>single');
    qFeat= reshape(qFeat, [], db.numQueries);
    if delFile, delete(qFeatFn); end
    
    dbFeatFn= sprintf('%s_db.bin', outPrefix);
    tmpFn= sprintf('%s.tmp', dbFeatFn);
    serialAllFeats(net, db.dbPath, db.dbImageFns, ...
        tmpFn, 'useGPU', opts.useGPU, 'batchSize', opts.computeBatchSize);
    movefile(tmpFn, dbFeatFn);
    
    dbFeat= fread( fopen(dbFeatFn, 'rb'), [size(qFeat,1), db.numImages], 'float32=>single');
    if delFile, delete(dbFeatFn); end
end

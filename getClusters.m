function clsts= getClusters(net, opts, clstFn, k, dbTrain, trainDescFn)
    
    if ~exist(clstFn, 'file')
        
        if ~exist(trainDescFn, 'file')
            
            simpleNnOpts= {'conserveMemory', true, 'mode', 'test'};
            
            if opts.useGPU
                net= relja_simplenn_move(net, 'gpu');
            end
            
            % ---------- extract training descriptors
            
            relja_display('Computing training descriptors');
            
            nTrain= 50000;
            nPerImage= 100;
            nIm= ceil(nTrain/nPerImage);
            
            rng(43);
            trainIDs= randsample(dbTrain.numImages, nIm);
            
            nTotal= 0;
            
            prog= tic;
            
            for iIm= 1:nIm
                relja_progress(iIm, nIm, 'extract train descs', prog);
                
                % --- extract descriptors
                
                % didn't want to complicate with batches here as it's only done once (per network and training set)
                
                im= vl_imreadjpeg({[dbTrain.dbPath, dbTrain.dbImageFns{iIm}]});
                im= im{1};
                
                % fix non-colour images
                if size(im,3)==1
                    im= cat(3,im,im,im);
                end
                
                im(:,:,1)= im(:,:,1) - net.meta.normalization.averageImage(1,1,1);
                im(:,:,2)= im(:,:,2) - net.meta.normalization.averageImage(1,1,2);
                im(:,:,3)= im(:,:,3) - net.meta.normalization.averageImage(1,1,3);
                
                if opts.useGPU
                    im= gpuArray(im);
                end
                
                res= vl_simplenn(net, im, [], [], simpleNnOpts{:});
                descs= gather(res(end).x);
                descs= reshape( descs, [], size(descs,3) )';
                
                % --- sample descriptors
                
                nThis= min( min(nPerImage, size(descs,2)), nTrain - nTotal );
                descs= descs(:, randsample( size(descs,2), nThis ) );
                
                if iIm==1
                    trainDescs= zeros( size(descs,1), nTrain, 'single' );
                end
                
                trainDescs(:, nTotal+[1:nThis])= descs;
                nTotal= nTotal+nThis;
            end
            
            trainDescs= trainDescs(:, 1:nTotal);
            
            % move back to CPU addLayers() assumes it
            if opts.useGPU
                net= relja_simplenn_move(net, 'cpu');
            end
            
            save(trainDescFn, 'trainDescs');
        else
            relja_display('Loading training descriptors');
            load(trainDescFn, 'trainDescs');
        end
        
        % ---------- Cluster descriptors
        
        relja_display('Computing clusters');
        clsts= yael_kmeans(trainDescs, k, 'niter', 100, 'verbose', 0, 'seed', 43);
        clear trainDescs;
        
        save(clstFn, 'clsts');
    else
        relja_display('Loading clusters');
        load(clstFn, 'clsts');
        assert(size(clsts, 2)==k);
    end
    
end

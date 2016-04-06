%  Uses the network `net` to extract image representations from a list
%  of image filenames `imageFns`.
%  `imageFns` is a cell array containing image file names relative
%  to the `imPath` (i.e. `[imPath, imageFns{i}]` is a valid JPEG image).
%  The representations are saved to `outFn` (single 4-byte floats).
%
%  Additional options:
%
%  `useGPU': Use the GPU or not
%
%  `batchSize': The number of images to process in a batch. Note that if your
%       input images are not all of same size (they are in place recognition
%       datasets), you should set `batchSize` to 1.

function serialAllFeats(net, imPath, imageFns, outFn, varargin)
    
    opts= struct(...
        'useGPU', true, ...
        'numThreads', 12, ...
        'batchSize', 10 ...
        );
    opts= vl_argparse(opts, varargin);
    simpleNnOpts= {'conserveMemory', true, 'mode', 'test'};
    
    relja_display('serialAllFeats: Start');
    
    net= netPrepareForTest(net);
    
    if opts.useGPU
        net= relja_simplenn_move(net, 'gpu');
    else
        net= relja_simplenn_move(net, 'cpu');
    end
    
    nImages= length(imageFns);
    nBatches= ceil( nImages / opts.batchSize );
    
    numInBuffer= 0;
    
    prog= tic;
    
    for iBatch= 1:nBatches
        relja_progress(iBatch, nBatches, 'serialAllFeats', prog);
        
        iStart= (iBatch-1)*opts.batchSize +1;
        iEnd= min(iStart + opts.batchSize-1, nImages);
        
        thisImageFns= strcat( imPath, imageFns(iStart:iEnd) );
        thisNumIms= iEnd-iStart+1;
        
        ims_= vl_imreadjpeg(thisImageFns, 'numThreads', opts.numThreads);
        
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
        
        if opts.useGPU
            ims= gpuArray(ims);
        end
        
        % ---------- extract features
        res= vl_simplenn(net, ims, [], [], simpleNnOpts{:});
        clear ims;
        feats= reshape( gather(res(end).x), [], thisNumIms );
        clear res;
        
        if iBatch==1
            fout= fopen(outFn, 'wb');
            [buffer, bufferSize]= relja_makeBuffer(feats(:,1), 200);
        end
        
        if (numInBuffer + thisNumIms > bufferSize)
            fwrite(fout, buffer(:, 1:numInBuffer), class(buffer));
            numInBuffer= 0;
        end
        buffer(:, numInBuffer+[1:thisNumIms])= feats;
        numInBuffer= numInBuffer+thisNumIms;
        
    end
    
    %%%%% flush buffer
    
    if numInBuffer>0
        fwrite(fout, buffer(:, 1:numInBuffer), class(buffer));
    end
    fclose(fout);
    
    
    relja_display('serialAllFeats: Done');
end

%  Computes an image representation given a network (such as NetVLAD).
%  NOTE: If you are processing more images of the same size
%  (e.g. like we do cutouts from Street View panoramas), it is MUCH faster
%  to process them in batches using `serialAllFeats`.
%  Even if the images are not of the same size, use `serialAllFeats`
%  (with `batchSize` set to 1) as it avoids copying the potentially big
%  network back and forward between the CPU and the GPU.
%  This function is mostly provided for completeness and for a quick start,
%  but try to use `serialAllFeats` in general.
%
%  The function simply performs a forward pass on the input image.
%  The image should be single in the range 0-255, as returned by running vl_imreadjpeg:
%
%  im= vl_imreadjpeg({imageFn}); im= im{1};
%
%  The same can usually be achieved with single(imread(imageFn)); but this is
%  **NOT** recommended as it is not guaranteed to produce a 0-255 output
%  (e.g. if the image is binary, or 16-bit colours are used, etc)
%
%  Additional options:
%
%  `useGPU': Use the GPU or not

function feats= computeRepresentation(net, im, varargin)
    
    if ~isa(im, 'single')
        if ~isa(im, 'uint8')
            error('The image is not in the correct format, type `help computeFeatures` for details');
        end
        warning('The input image was not single -- converting to single but it is likely there is something else wrong as well, type `help computeFeatures` for details');
        im= single(im);
    end
    
    opts= struct(...
        'useGPU', true ...
        );
    opts= vl_argparse(opts, varargin);
    simpleNnOpts= {'conserveMemory', true, 'mode', 'test'};
    
    net= netPrepareForTest(net);
    
    if opts.useGPU
        net= relja_simplenn_move(net, 'gpu');
    else
        net= relja_simplenn_move(net, 'cpu');
    end
    
    if size(im,3)==1
        im= cat(3,im,im,im);
    end
    
    im(:,:,1)= im(:,:,1) - net.meta.normalization.averageImage(1,1,1);
    im(:,:,2)= im(:,:,2) - net.meta.normalization.averageImage(1,1,2);
    im(:,:,3)= im(:,:,3) - net.meta.normalization.averageImage(1,1,3);
    
    if opts.useGPU
        im= gpuArray(im);
    end
    
    % ---------- extract features
    res= vl_simplenn(net, im, [], [], simpleNnOpts{:});
    clear im;
    feats= reshape( gather(res(end).x), [], 1 );
    clear res;
end

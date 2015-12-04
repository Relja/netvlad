function [C, I]= yael_kmeans(v, k, varargin)
    opts= struct(...
        'redo', 1, ...
        'verbose', 1, ...
        'seed', 0, ...
        'init', 1, ...
        'niter', 50 ...
        );
    
    opts= vl_argparse(opts, varargin);
    
    if opts.init==1
        matInit= 'sample';
    else
        matInit= 'plus';
    end
    
    if opts.verbose>1
        matVerb= 'iter';
    elseif opts.verbose==1
        matVerb= 'final';
    else
        matVerb= 'off';
    end
    
    vl_twister('state', opts.seed);
    
    [I, C]= kmeans(v', k, 'Start', matInit, 'Replicates', opts.redo, 'MaxIter', opts.niter, 'Display', matVerb);
    C= C';
    I= I;
    
    I= int32(I);
end

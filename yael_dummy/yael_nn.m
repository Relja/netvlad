function [ids, dis]= yael_nn(v, q, k, distype)
    assert( nargin<4 || distype==2 );
    if nargin<3, k= 1; end
    assert(k<size(v,2));
    
    ids= zeros(k, size(q,2), 'int32');
    dis= zeros(k, size(q,2), 'single');
    
    for iVec= 1:size(q,2)
        ds= sum( bsxfun(@minus, v, q(:,1)).^2, 1 );
        [ds, inds]= sort(ds);
        dis(:,iVec)= ds(1:k);
        ids(:,iVec)= inds(1:k);
    end
end

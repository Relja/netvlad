function [ids, ds]= rawNnSearch(q, db, k)
    % wrapper as yael_nn insists to return two values while I sometimes need just one (for anonymous functions..)
    if nargin<3, k=1; end
    k= min(k, size(db,2));
    [ids, ds]= yael_nn(db, q, k);
end

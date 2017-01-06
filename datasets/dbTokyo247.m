classdef dbTokyo247 < dbBase
    
    methods
    
        function db= dbTokyo247()
            db.name= sprintf('tokyo247');
            
            paths= localPaths();
            dbRoot= paths.dsetRootTokyo247;
            db.dbPath= [dbRoot, 'images/'];
            db.qPath= [dbRoot, 'query/'];
            
            db.dbLoad();
        end
        
        function [ids, ds]= nnSearchPostprocess(db, searcher, iQuery, nTop)
            % perform non-max suppression like in Torii et al. CVPR 2015
            [ids, ds]= searcher(iQuery, nTop*12); % 12 cutouts per panorama
            [~, uniqInd, ~]= unique( idivide(ids-1, 12) ,'stable');
            uniqInd= uniqInd(1:min(end,nTop));
            ids= ids(uniqInd);
            ds= ds(uniqInd);
        end
        
    end
    
end


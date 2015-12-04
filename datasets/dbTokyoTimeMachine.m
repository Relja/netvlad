classdef dbTokyoTimeMachine < dbBase
    
    properties
        dbTimeStamp, qTimeStamp
    end
    
    methods
    
        function db= dbTokyoTimeMachine(whichSet)
            % whichSet is one of: train, val
            
            assert( ismember(whichSet, {'train', 'val'}) );
            
            db.name= sprintf('tokyoTM_%s', whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetRootTokyoTM;
            db.dbPath= [dbRoot, 'images/'];
            db.qPath= [dbRoot, 'images/'];
            
            db.dbLoad();
        end
        
        function posIDs= nontrivialPosQ(db, iQuery)
            [posIDs, dSq]= db.cp.getPosIDs(db.utmQ(:,iQuery));
            posIDs= posIDs(dSq>1 & dSq<=db.nonTrivPosDistSqThr & db.qTimeStamp(iQuery)~=db.dbTimeStamp(posIDs) );
        end
        
        function posIDs= nontrivialPosDb(db, iDb)
            [posIDs, dSq]= db.cp.getPosDbIDs(iDb);
            posIDs= posIDs(dSq>1 & dSq<=db.nonTrivPosDistSqThr & db.dbTimeStamp(iDb)~=db.dbTimeStamp(posIDs) );
        end
        
        function [ids, ds]= nnSearchPostprocess(db, searcher, iQuery, nTop)
            % there's roughly up to 5 panoramas at different times per location, so a very conservative estimate..
            [ids, ds]= searcher(iQuery, nTop*10);
            keep= db.qTimeStamp(iQuery)~=db.dbTimeStamp(ids);
            ids= ids(keep);
            ds= ds(keep);
            ids= ids(1:min(end,nTop));
            ds= ds(1:min(end,nTop));
        end
        
    end
    
end


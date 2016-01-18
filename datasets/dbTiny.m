classdef dbTiny < dbBase
    
    methods
    
        function db= dbTiny(whichSet)
            % whichSet is one of: train, val
            
            db_= dbTokyoTimeMachine(whichSet);
            
            for propName= properties(db)'
                db.(propName{1})= db_.(propName{1});
            end
            clear db_;
            db.name= sprintf('tokyoTinyTM_%s', whichSet);
            
            % construct a tiny dataset
            rng(43);
            dbIms= [];
            qIms= [];
            while length(dbIms)<120
                thisIms= find( sum( bsxfun(@minus, db.utmDb, db.utmDb(:, randsample(db.numImages,1) )).^2, 1 ) < 5 );
                if length(thisIms) > 12
                    qIms= [qIms, thisIms(1:6)];
                    dbIms= [dbIms, thisIms(12+[1:12])];
                end
            end
            
            db.utmQ= db.utmDb(:, qIms);
            db.utmDb= db.utmDb(:, dbIms);
            db.qImageFns= db.dbImageFns(qIms);
            db.dbImageFns= db.dbImageFns(dbIms);
            db.numQueries= length(qIms);
            db.numImages= length(dbIms);
            db.cp= closePosition(db.utmDb, db.posDistThr);
            
        end
        
        function posIDs= nontrivialPosQ(db, iQuery)
            [posIDs, dSq]= db.cp.getPosIDs(db.utmQ(:,iQuery));
            posIDs= posIDs(dSq<=db.nonTrivPosDistSqThr );
        end
        
        function posIDs= nontrivialPosDb(db, iDb)
            [posIDs, dSq]= db.cp.getPosDbIDs(iDb);
            posIDs= posIDs( dSq<=db.nonTrivPosDistSqThr );
        end
        
    end
    
end


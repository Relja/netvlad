% Base class for the dataset specification
%
% To make your own dataset (see dbPitts.m for an easy example):
% 1. Inherit from dbBase
% 2. In the constructor, set a short identifier of the dataset in db.name
% 3. Save a matlab structure called dbStruct to <paths.dsetSpecDir>/<db.name>.mat, which contains:
%   - dbImageFns: cell array of database image file names relative to dataset root
%   - qImageFns: cell array of query image file names relative to dataset root
%   - utmDb:  2x <number_of_database_images>, containing (x,y) UTM coordinates for each database image
%   - utmQ:  2x <number_of_query_images>, containing (x,y) UTM coordinates for each query image
%   - posDistThr: distance in meters which defines potential positives
%   - nonTrivPosDistSqThr: squared distance in meters which defines the potential positives used for training
% 4. In the constructor, set db.dbPath and db.qPath specifying the root locations of database and query images, respectively. Presumably, like in dbPitts.m, you want to load these from a configuration file. The variables should be such that [db.dbPath, dbImageFns{i}] and [db.qPath, qImageFns{i}] form the full paths to database/query images.
% 5. Finally: call db.dbLoad(); at the end of the constructor
% 6. Optionally: you can override the methods for some more functionality, e.g. for Tokyo Time Machine we modify the fuction nontrivialPosQ which gets all potential positives for a query that are non-trivial (don't come from the same panorama). For Time Machine data, we also make sure that the nontrivial potential positives are taken at different times than the query panorama (for generalization, c.f. our NetVLAD paper). There was no need for this for the Pittsburgh dataset as the query and the database sets were taken at different times, but for TokyoTM the query set is constructed out of the database set. Furthermore, one can also supplu 'nnSearchPostprocess' which filters search results (used in testCore.m), e.g. it is done for Tokyo 24/7 to follow the standard test procedure for this dataset (i.e. perform very simple non-max suppression)



classdef dbBase < handle
    
    properties
        name
        whichSet
        
        dbPath, dbImageFns, utmDb
        qPath, qImageFns, utmQ
        numImages, numQueries
        
        posDistThr, posDistSqThr
        nonTrivPosDistSqThr
        
        cp
    end
    
    methods
    
        function dbLoad(db)
            
            % load custom information
            
            paths= localPaths();
            dbFn= sprintf('%s/%s.mat', paths.dsetSpecDir, db.name);
            
            if exist(dbFn, 'file')
                load(dbFn, 'dbStruct');
                for propName= fieldnames(dbStruct)'
                    if ~ismember(propName{1}, {'cp'})
                        db.(propName{1})= dbStruct.(propName{1});
                    end
                end
                clear dbStruct propName;
            else
                error('Download the database file (%s.mat) and set the correct dsetSpecDir in localPaths.m', db.name);
            end
            
            % generate other useful data
            
            db.cp= closePosition(db.utmDb, db.posDistThr);
            db.numImages= length(db.dbImageFns);
            db.numQueries= length(db.qImageFns);
            assert( size(db.utmDb, 2) == db.numImages );
            assert( size(db.utmQ, 2) == db.numQueries );
            db.posDistSqThr= db.posDistThr^2;
            
            % make paths absolute just in case (e.g. vl_imreadjpeg needs absolute path)
            
            for propName= properties(db)'
                s= db.(propName{1});
                if isstr(s)
                    db.(propName{1})= relja_expandUser(s);
                end
            end
            
        end
        
        function isP= isPosQ(db, iQuery, iDb)
            isP= db.cp.isPos( db.utmQ(:,iQuery), iDb );
        end
        
        function isP= isPosDb(db, iDbQuery, iDb)
            isP= db.cp.isPos( db.utmDb(:,iDbQuery), iDb );
        end
        
        function posIDs= posDb(db, iDb)
            posIDs= db.cp.getPosDbIDs(iDb);
        end
        
        function negIDs= sampleNegsQ(db, iQuery, n)
            negIDs= db.sampleNegs(db.utmQ(:,iQuery), n);
        end
        
        function negIDs= sampleNegsDb(db, iDb, n)
            negIDs= db.sampleNegs(db.utmDb(:,iDb), n);
        end
        
        function negIDs= sampleNegs(db, utm, n)
            negIDs= [];
            
            while length(negIDs) < n
                negs= randsample(db.numImages, round(n*1.1));
                
                isNeg= ~db.cp.isPos(utm, negs);
                nNewNeg= sum(isNeg);
                
                if nNewNeg>0
                    negIDs= unique([negIDs; negs(isNeg)]);
                end
            end
            
            if length(negIDs) > n
                negIDs= randsample(negIDs, n);
            end
            
        end
        
        function posIDs= nontrivialPosQ(db, iQuery)
            [posIDs, dSq]= db.cp.getPosIDs(db.utmQ(:,iQuery));
            posIDs= posIDs(dSq>1 & dSq<=db.nonTrivPosDistSqThr );
        end
        
        function posIDs= nontrivialPosDb(db, iDb)
            [posIDs, dSq]= db.cp.getPosDbIDs(iDb);
            posIDs= posIDs(dSq>1 & dSq<=db.nonTrivPosDistSqThr );
        end
    end
    
end


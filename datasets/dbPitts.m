classdef dbPitts < dbBase
    
    methods
    
        function db= dbPitts(fullSize, whichSet)
            % fullSize is: true or false
            % whichSet is one of: train, val, test
            
            assert( ismember(whichSet, {'train', 'val', 'test'}) );
            
            if fullSize
                sizeSuffix= '250k';
            else
                sizeSuffix= '30k';
            end
            
            db.name= sprintf('pitts%s_%s', sizeSuffix, whichSet);
            
            paths= localPaths();
            dbRoot= paths.dsetRootPitts;
            db.dbPath= [dbRoot, 'images/'];
            db.qPath= [dbRoot, 'queries/'];
            
            db.dbLoad();
        end
        
    end
    
end


classdef closePosition < handle
    
    properties
        utm
        limL
        limU
        posDistSqThr
        delta
        hashTab
        hashSz
    end
    
    methods
        
        function cp= closePosition(utm, posDistThr)
            cp.utm= utm;
            cp.limL= min(utm,[],2)-1;
            cp.limU= max(utm,[],2)+1;
            cp.posDistSqThr= posDistThr^2;
            cp.delta= posDistThr+1;
            
            cp.hashTab= cell( ...
                ceil( (cp.limU(1)-cp.limL(1)) / cp.delta ), ...
                ceil( (cp.limU(2)-cp.limL(2)) / cp.delta ) );
            cp.hashSz= size(cp.hashTab);
            
            inds= cp.getInds(cp.utm);
            for iPos= 1:size(cp.utm, 2)
                cp.hashTab{ inds(1, iPos), inds(2, iPos) }= [ ...
                    cp.hashTab{ inds(1, iPos), inds(2, iPos) }; ...
                    iPos];
            end
        end
        
        function inds= getInds(cp, utm)
            inds= ceil( bsxfun(@minus, utm, cp.limL) / cp.delta );
        end
        
        function [posIDs, ds]= getPosIDs(cp, utm)
            inds= cp.getInds( utm );
            posIDs= cat(1, ...
                cp.hashTab{ ...
                    max(inds(1)-1, 1) : min(inds(1)+1, cp.hashSz(1)), ...
                    max(inds(2)-1, 1) : min(inds(2)+1, cp.hashSz(2)) } ...
                );
            
            ds= sum( bsxfun(@minus, utm, cp.utm(:,posIDs)) .^ 2, 1 );
            keep= ds <= cp.posDistSqThr;
            posIDs= posIDs(keep);
            ds= ds(keep);
        end
        
        function [posIDs, ds]= getPosDbIDs(cp, dbID)
            [posIDs, ds]= cp.getPosIDs(cp.utm(:,dbID));
        end
        
        function isP= isPos(cp, utm, IDs)
            isP= ismember(IDs, cp.getPosIDs(utm));
        end
        
    end
    
end

classdef layerWholeL2Normalize
    
    properties
        type= 'custom'
        name= 'wholeL2'
        precious= false
    end
    
    methods
        
        function l= layerWholeL2Normalize(name)
            if nargin>0, l.name= name; end
        end
        
        function y= forward_(p, x)
            
            batchSize= size(x, 4);
            y= relja_l2normalize_col( reshape(x, [], batchSize) );
            y= reshape(y, [1,1,size(y,1), batchSize]);
        end
        
        function dzdx= backward_(p, x, dzdy)
            batchSize= size(x, 4);
            xr= reshape(x, [], batchSize);
            dzdy= reshape(dzdy, [], batchSize);
            xNorm= sqrt(sum(xr.^2,1)) + 1e-12;
            % dim= size(xr, 1);
            
            % D: d(yi)/d(xj)= (i==j)/xnorm - xi*xj / xnorm^3
            % where xnorm= sqrt(sum(x.^2))
            % dzdx= D' * dzdy
            % batchSize==1: D= -xr/(xNorm^3) * xr' + 1/xNorm*eye(dim);
            % for speed better:
            % -xr*(xr'*dzdx)/xNorm^3 + dzdx/xNorm
            
            dzdx= xr; % just to create the matrix, will be overwritten
            
            for iB= 1:batchSize
                % Slow:
                % dzdx(:,iB)= ...
                %     ( -xr(:,iB)* xr(:,iB)'/(xNorm(iB)^3) + eye(dim)/xNorm(iB) ) ...
                %     * dzdy(:,iB);
                % Fast:
                dzdx(:,iB)= ...
                   -xr(:,iB)* (xr(:,iB)'*dzdy(:,iB))/(xNorm(iB)^3) + dzdy(:,iB)/xNorm(iB);
            end
            
            dzdx= reshape(dzdx, size(x));
        end
        
    end
    
    methods (Static)
        
        function res1= forward(p, res0, res1)
            res1.x= p.forward_(res0.x);
        end
        
        function res0= backward(p, res0, res1)
            res0.dzdx= p.backward_(res0.x, res1.dzdx);
        end
    
    end
    
end


% Can implement with pure matconvnet + reshapes, but my implementation seems to be much faster (at least for in my setup / usecase)

%  classdef layerWholeL2Normalize < handle
%      
%      properties
%          type= 'custom'
%          name= 'wholeL2'
%      end
%      
%      methods
%          
%          function l= layerWholeL2Normalize(name)
%              if nargin>0, l.name= name; end
%          end
%          
%      end
%      
%      methods (Static)
%          
%          function res1= forward(l, res0, res1)
%              D= relja_numel(res0.x, [1,2,3]);
%              batchSize= size(res0.x, 4);
%              res1.x= vl_nnnormalize( ...
%                  reshape(res0.x, [1, 1, D, batchSize]), ...
%                  [2*D, 1e-12, 1, 0.5]);
%          end
%          
%          function res0= backward(l, res0, res1)
%              D= relja_numel(res0.x, [1,2,3]);
%              batchSize= size(res0.x, 4);
%              res0.dzdx= reshape( vl_nnnormalize( ...
%                      reshape(res0.x, [1, 1, D, batchSize]), ...
%                      [2*D, 1e-12, 1, 0.5], ...
%                      reshape(res1.dzdx, [1, 1, D, batchSize])), ...
%                      size(res0.x) );
%          end
%      
%      end
%      
%  end

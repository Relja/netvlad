classdef layerTotalAvgPool
    
    properties
        type= 'custom'
        name= 'avgTotal'
        precious= false
    end
    
    methods
        
        function l= layerTotalAvgPool(name)
            if nargin>0, l.name= name; end
        end
        
    end
    
    methods (Static)
        
        function res1= forward(p, res0, res1)
            res1.x= vl_nnpool(res0.x, [size(res0.x,1), size(res0.x,2)], ...
                'method', 'avg');
        end
        
        function res0= backward(p, res0, res1)
            res0.dzdx= vl_nnpool(res0.x, [size(res0.x,1), size(res0.x,2)], res1.dzdx, ...
                'method', 'avg');
        end
    
    end
    
end

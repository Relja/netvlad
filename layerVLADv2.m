classdef layerVLADv2
    
    properties
        type= 'custom'
        name= 'VLAD'
        K
        D
        vladDim
        weights
        momentum
        learningRate
        weightDecay
        precious= false
    end
    
    methods
        
        function l= layerVLADv2(name)
            if nargin>0, l.name= name; end
        end
        
        function l= constructor(l, weights)
            % weights: DxK (assignment weight alpha*2*clst ), 1xK (alpha*|clsts|^2), DxK (-clst)
            assert( length(weights)==3 );
            assert( size(weights{2},1) == 1 );
            assert( size(weights{1},2) == size(weights{2},2) );
            assert( size(weights{1},1) == size(weights{3},1) );
            assert( size(weights{1},2) == size(weights{3},2) );
            l.D= size(weights{1}, 1);
            l.K= size(weights{1}, 2);
            l.vladDim= l.D*l.K;
            
            l.weights= { reshape( weights{1}, [1,1,l.D,l.K]), ...
                         weights{2}, ...
                         reshape( weights{3}, [1,1,l.D,l.K]) };
        end
        
        function y= forward_(l, x)
            
            batchSize= size(x, 4);
            
            % --- assign
            
            assgn= vl_nnsoftmax( vl_nnconv(x, l.weights{1}, l.weights{2}) );
            
            % --- aggregate
            
            if isa(x, 'gpuArray')
                y= zeros([1, l.K, l.D, batchSize], 'single', 'gpuArray');
            else
                y= zeros([1, l.K, l.D, batchSize], 'single');
            end
            
            for iK= 1:l.K
                % --- sum over descriptors: assignment_iK * (descs - offset_iK)
                y(:,iK,:,:)= ...
                    sum(sum( ...,
                        repmat( assgn(:,:,iK,:), [1,1,l.D,1] ) .* ...
                        vl_nnconv(x, [], l.weights{3}(1,1,:,iK)), ...
                        1), 2);
                % % I expected this to be faster, but it's not:
                % y(:,iK,:,:)= ...
                %     sum(sum( ...,
                %         bsxfun(@times, ...
                %             assgn(:,:,iK,:), ...
                %             vl_nnconv(x, [], l.weights{3}(1,1,:,iK)) ...
                %         ), 1), 2);
            end
            
            % --- normalizations (intra-normalization, L2 normalization)
            % performed outside as separate layers
            
        end
        
        function [dzdx, dzdw]= backward_(l, x, dzdy)
            
            batchSize= size(x, 4);
            H= size(x, 1);
            W= size(x, 2);
            % assert(l.D==size(x, 3));
            
            % TODO: stupid to run forward again? remember results?
            
            % --- assign
            
            p= vl_nnconv(x, l.weights{1}, l.weights{2});
            assgn= vl_nnsoftmax(p);
            
            % --- dz/da (soft assignment)
            
            dzda= assgn; % just for the shape/class
            
            for iK= 1:l.K
                dzda(:,:,iK,:)= sum( ...
                        bsxfun(@times, ...
                            dzdy(:,iK,:,:), ...
                            vl_nnconv(x, [], l.weights{3}(1,1,:,iK))), ...
                        3);
            end
            
            % --- dz/dp ("distance")
            
            dzdp= vl_nnsoftmax(p, dzda); clear dzda p;
            
            % --- dz/dw1 (assignment clusters) and dz/dx (via assignment)
            
            [dzdx, dzdw{1}, dzdw{2}]= vl_nnconv(x, l.weights{1}, l.weights{2}, dzdp); clear dzdp;
            
            % --- dz/dx (via aggregation)
            % --- and add to current dz/dx to get the full thing
            
            dzdy= reshape(dzdy, [l.K, l.D, batchSize]);
            
            assgn_= reshape(assgn, [H*W, l.K, batchSize]);
            for iB= 1:batchSize
                dzdx(:,:,:,iB)= dzdx(:,:,:,iB) + reshape( ...
                    assgn_(:,:,iB) * dzdy(:,:,iB), ...
                    [H, W, l.D]);
            end
            clear assgn_;
            
            % --- dz/dw2 (offset)
            
            dzdw{3}= reshape( sum( ...
                dzdy .* ...
                repmat( ...
                    reshape( sum(sum(assgn,1),2), [l.K, 1, batchSize] ), ...
                    [1, l.D, 1] ), ...
                3 )', [1, 1, l.D, l.K] );
            
        end
        
        function objStruct= saveobj(obj)
            objStruct= relja_saveobj(obj);
        end
        
    end
    
    methods (Static)
        
        function res1= forward(l, res0, res1)
            res1.x= l.forward_(res0.x);
        end
        
        function res0= backward(l, res0, res1)
            [res0.dzdx, res0.dzdw]= l.backward_(res0.x, res1.dzdx);
        end
        
        function l= loadobj(objStruct)
            l= layerVLADv2();
            l= relja_loadobj(l, objStruct);
        end
    
    end
    
end

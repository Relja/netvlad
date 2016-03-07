function epoch= getBestEpoch(recalls, recallNs, N)
    if nargin<3, N= 5; end
    
    % This command is generally OK
    % [~, epoch]= max(recalls(recallNs==N,:));
    % but the following does tie-breaking
    
    nRecalls= length(recallNs);
    nEpochs= size(recalls, 2);
    assert( nRecalls==size(recalls, 1) );
    
    posN= find( recallNs==N, 1);
    assert(~isempty(posN));
    posNs= posN:-1:1;
    if posN<nRecalls
        posNs= [posNs, (posN+1):nRecalls];
    end
    
    potential= true(1, nEpochs);
    for posN= posNs
        maxVal= max(recalls(posN, potential));
        isMax= abs(recalls(posN,:)-maxVal)<1e-6;
        potential= potential & isMax;
        if sum(potential)<=1
            break;
        end
    end
    
    epoch= find( potential, 1 );
    assert(~isempty(epoch));
end

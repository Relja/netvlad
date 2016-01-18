% Upgrades the network object to the most recent format.
% The motivation is the same as the motivation behind vl_simplenn_tidy
% (which is also executed during the call) as it enables NetVLAD code
% to evolve while not breaking backward compatibility

function net= relja_simplenn_tidy(net)
    
    % --- NetVLAD save values which will be deleted by MatConvNet tidy
    
    toSave= struct();
    
    % from v100-v101
    fieldNames= {'netID', 'sessionID', 'epoch'};
    
    for fieldName_= fieldNames
        fieldName= fieldName_{1};
        if isfield(net, fieldName)
            toSave.(fieldName)= net.(fieldName);
            net= rmfield(net, fieldName);
        end
    end
    
    % --- MatConvNet tidy
    
    net= vl_simplenn_tidy(net);
    
    % --- NetVLAD tidy
    
    for fieldName_= fieldNames
        fieldName= fieldName_{1};
        if isfield(toSave, fieldName)
            net.meta.(fieldName)= toSave.(fieldName);
            toSave= rmfield(toSave, fieldName);
        end
    end
    assert(isempty(fieldnames(toSave)));
    
end

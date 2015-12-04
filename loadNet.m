function net= loadNet(netID, layerName)
    if nargin<2, layerName= '_relja_none_'; end
    
    switch netID
        case 'vd16'
            netname= 'imagenet-vgg-verydeep-16.mat';
        case 'vd19'
            netname= 'imagenet-vgg-verydeep-19.mat';
        case 'caffe'
            netname= 'imagenet-caffe-ref.mat';
        case 'places'
            netname= 'places-caffe.mat';
        otherwise
            error( 'Unknown network ID', netID );
    end
    
    paths= localPaths();
    net= load( fullfile(paths.pretrainedCNNs, netname));
    
    if isfield(net, 'classes')
        net= rmfield(net, 'classes');
    end
    
    if ~strcmp(layerName, '_relja_none_')
        net= relja_cropToLayer(net, layerName);
        layerNameStr= ['_', layerName];
    else
        layerNameStr= '';
    end
    net= relja_swapLayersForEfficiency(net);
    net.netID= netID;
    
    net.sessionID= sprintf('%s_offtheshelf%s', netID, layerNameStr);
    net.epoch= 0;
    
end

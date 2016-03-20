%  Author: Relja Arandjelovic (relja@relja.info)

% This file contains a few examples of how to test our NetVLAD on standard object/image retrieval benchmarks, refer to README.md for setup instructions, and to our project page for all relevant information (e.g. our paper): http://www.di.ens.fr/willow/research/netvlad/


%  The code samples use the GPU by default, if you want to use the CPU instead (quite slow!), add `'useGPU', false` to the affected function call (serialAllFeats)



% Set the MATLAB paths
setup;



% ---------- Use/test our networks

% Load our network
netID= 'vd16_pitts30k_conv5_3_vlad_preL2_intra_white';
paths= localPaths();
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);



% --- Oxford Buildings / Paris Buildings

dsetName= 'ox5k'; % change to `'paris'` for Paris Buildings

% Define if you want to use the query ROI information.
% The original evaluation procedure for Oxford / Paris DOES use ROIs,
% so we really recommend to keep this as is, but we also include the setting
% where they are not used in order to be able to compare to some recent works.
% Please continue using ROIs as this makes it easier to compare numbers
% accross different papers, and it is the proper evaluation procedure for
% these datasets.
% Note that, for ease of implementation, if you are using ROIs the code will
% crop the query images and save them into a new folder
% `images_crop_<amount_of_crop>` inside the dataset root
% (i.e. makes sure this is writable). The crop is made by extending the ROI
% by half of the receptive field size, as this ensures that the centres of
% all 'features' % (i.e. conv responses) are inside the ROI; this is a setting
% similar to the original test procedure where a SIFT feature is kept iff its
% centre is inside the ROI. Note that this is ROI extension is only valid
% for single-scale extraction of features, if you want multiple scales then
% the implementation needs to be different
%
% To use the ROI: specify the receptive field size (can be 0 for a tight crop)
% Not to use the ROI (not recommended): specify a negative number

useROI= true;
if useROI
    lastConvLayer= find(ismember(relja_layerTypes(net), 'custom'),1)-1; % Relies on the fact that NetVLAD, Max and Avg pooling are implemented as a custom layer and are the first custom layer in the network. Change if you use another network which has other custom layers before
    netBottom= net;
    netBottom.layers= netBottom.layers(1:lastConvLayer);
    info= vl_simplenn_display(netBottom);
    clear netBottom;
    recFieldSize= info.receptiveFieldSize(:, end);
    assert(recFieldSize(1) == recFieldSize(2));% we are assuming square receptive fields, otherwise dbVGG needs to change to account for non-square
    recFieldSize= recFieldSize(1);
    strMode= 'crop';
else
    recFieldSize= -1;
    strMode= 'full';
end

dbTest= dbVGG(dsetName, recFieldSize);

% Compute db image representations (images have different resolutions so batchSize is constrained to 1)
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1);

if useROI
    qFeatFn = sprintf('%s%s_%s_q_%d.bin', paths.outPrefix, netID, dbTest.name, recFieldSize);
    serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1);
end

% Load the image representations
dbFeat= fread( fopen(dbFeatFn, 'rb'), inf, 'float32=>single');
dbFeat= reshape(dbFeat, [], dbTest.numImages);
nDims= size(dbFeat, 1);
if useROI
    qFeat= fread( fopen(qFeatFn, 'rb'), [nDims, dbTest.numImages], 'float32=>single');
    assert(size(qFeat,2)==dbTest.numQueries);
else
    qFeat= dbFeat(:, dbTest.queryIDs);
end

% Measure recall@N
mAP= relja_retrievalMAP(dbTest, struct('db', dbFeat, 'qs', qFeat), true);
relja_display('Performance on %s (%s), %d-D: %.2f', dbTest.name, strMode, size(dbFeat,1), mAP*100);

% Try 256-D
D= 256;
mAPsmall= relja_retrievalMAP(dbTest, struct( ...
    'db', relja_l2normalize_col(dbFeat(1:D,:)), ...
    'qs', relja_l2normalize_col(qFeat(1:D,:)) ...
    ), true);
relja_display('Performance on %s (%s), %d-D: %.2f', dbTest.name, strMode, D, mAPsmall*100);



% --- Holidays

useRotated= false; % for rotated Holidays
dbTest= dbHolidays(useRotated); % it will automatically downscale images to (1024x768) pixels, as per the original testing procedure (see the Holidays website)

% Set the output filename for the database (query images are a subset)
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);

% Compute db image representations (images have different resolutions so batchSize is constrained to 1)
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 1);

% Load the image representations
dbFeat= fread( fopen(dbFeatFn, 'rb'), inf, 'float32=>single');
dbFeat= reshape(dbFeat, [], dbTest.numImages);
nDims= size(dbFeat, 1);
qFeat= dbFeat(:, dbTest.queryIDs);

mAP= relja_retrievalMAP(dbTest, struct('db', dbFeat, 'qs', qFeat), true);
relja_display('Performance on %s, %d-D: %.2f', dbTest.name, size(dbFeat,1), mAP*100);

% Try 256-D
D= 256;
mAPsmall= relja_retrievalMAP(dbTest, struct( ...
    'db', relja_l2normalize_col(dbFeat(1:D,:)), ...
    'qs', relja_l2normalize_col(qFeat(1:D,:)) ...
    ), true);
relja_display('Performance on %s, %d-D: %.2f', dbTest.name, D, mAPsmall*100);

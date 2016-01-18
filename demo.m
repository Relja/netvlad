%  Author: Relja Arandjelovic (relja@relja.info)

% This file contains a few examples of how to train and test CNNs for place recognition, refer to README.md for setup instructions, and to our project page for all relevant information (e.g. our paper): http://www.di.ens.fr/willow/research/netvlad/


%  The code samples use the GPU by default, if you want to use the CPU instead (very slow especially for training!), add `'useGPU', false` to the affected function calls (trainWeakly, addPCA, serialAllFeats, computeRepresentation)

% For a tiny example of running the training on a small dataset, which takes only a few minutes to run, refer to the end of this file.



% Set the MATLAB paths
setup;

error('Don''t run this script, it is only meant as a collection of useful commands');



% ---------- Use/test our networks

% Load our network
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= localPaths();
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );
net= relja_simplenn_tidy(net);

%  Compute the image representation by simply running the forward pass using
% the network `net` on the appropriately normalized image
% (see `computeRepresentation.m`).

im= vl_imreadjpeg({which('football.jpg')}); im= im{1}; % slightly convoluted because we need the full image path for `vl_imreadjpeg`, while `imread` is not appropriate - see `help computeRepresentation`
feats= computeRepresentation(net, im); % add `'useGPU', false` if you want to use the CPU

% To compute representations for many images, use the `serialAllFeats` function
% which is much faster as it uses batches and it moves the network to
% the GPU only once:
%
%          serialAllFeats(net, imPath, imageFns, outputFn);
%
%  `imageFns` is a cell array containing image file names relative to the `imPath` (i.e. `[imPath, imageFns{i}]` is a valid JPEG image), the representations are saved in binary format (single 4-byte floats). Batch size used for computing the forward pass can be changed by adding the `batchSize` parameter, e.g. `'batchSize', 10`. Note that if your input images are not all of same size (they are in place recognition datasets), you should set `batchSize` to 1.

%  To test the network on a place recognition dataset, set up the test dataset
dbTest= dbTokyo247();

% Set the output filenames for the database/query image representations
paths= localPaths();
dbFeatFn= sprintf('%s%s_%s_db.bin', paths.outPrefix, netID, dbTest.name);
qFeatFn = sprintf('%s%s_%s_q.bin', paths.outPrefix, netID, dbTest.name);

% Compute db/query image representations
serialAllFeats(net, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10); % adjust batchSize depending on your GPU / network size
serialAllFeats(net, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1

% Measure recall@N
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');



% ---------- Full train and test example: Tokyo
% Train: Tokyo Time Machine, Test: Tokyo 24/7

% Set up the train/val datasets
dbTrain= dbTokyoTimeMachine('train');
dbVal= dbTokyoTimeMachine('val');
lr= 0.0001;

% --- Train the VGG-16 network + NetVLAD, tuning down to conv5_1
sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_1', ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true);

% Get the best network
% This can be done even if training is not finished, it will find the best network so far
[~, bestNet]= pickBestNet(sessionID);

% Either use the above network as the image representation extractor (do: finalNet= bestNet), or do whitening (recommended):
finalNet= addPCA(bestNet, dbTrain, 'doWhite', true, 'pcaDim', 4096);

% --- Test

% Set up the test dataset
dbTest= dbTokyo247();

% Set the output filenames for the database/query image representations
paths= localPaths();
dbFeatFn= sprintf('%s%s_ep%06d_%s_db.bin', paths.outPrefix, finalNet.meta.sessionID, finalNet.meta.epoch, dbTest.name);
qFeatFn = sprintf('%s%s_ep%06d_%s_q.bin', paths.outPrefix, finalNet.meta.sessionID, finalNet.meta.epoch, dbTest.name);

% Compute db/query image representations
serialAllFeats(finalNet, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10); % adjust batchSize depending on your GPU / network size
serialAllFeats(finalNet, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1

% Measure recall@N
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');

% --- Test smaller dimensionalities:

% All that needs to be done (only valid for NetVLAD+whitening networks!)
% to reduce the dimensionality of the NetVLAD representation below 4096 to D
% is to keep the first D dimensions and L2-normalize.
% This is done automatically in `testFromFn` using the `cropToDim` option:

cropToDims= [64, 128, 256, 512, 1024, 2048, 4096];
recalls= [];
plotN= 5;
figure;

for iCropToDim= 1:length(cropToDims)
    cropToDim= cropToDims(iCropToDim);
    relja_display('D= %d', cropToDim);
    [recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn, [], 'cropToDim', cropToDim);
    
    whichRecall= find(opts.recallNs==plotN);
    recalls= [recalls, recall(whichRecall)];
    hold off;
    semilogx( cropToDims(1:iCropToDim), recalls, 'bo-');
    set(gca, 'XTick', cropToDims(1:iCropToDim));
    xlabel('Number of dimensions'); ylabel(sprintf('Recall@%d', plotN)); grid on;
    drawnow;
end



% ---------- Pittsburgh examples

% Pitts250k or Pitts30k?
doPitts250k= false;

if doPitts250k
    % Pittsburgh 250k
    lr= 0.0001;
else
    % Pittsburgh 30k
    lr= 0.001;
end

dbTrain= dbPitts(doPitts250k, 'train');
dbVal= dbPitts(doPitts250k, 'val');
dbTest= dbPitts(doPitts250k, 'test');

% Now just run the same code as above for Tokyo



% ---------- Miscellaneous

% --- Other potentially useful commands for training

% caffe vlad: backprop down to conv2

sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'caffe', 'layerName', 'conv5', 'backPropToLayer', 'conv2', ...
    'method', 'vlad_preL2_intra', ...
    'learningRate', lr, ...
    'doDraw', true);

% vgg16 max: backprop down to conv5

sessionID= trainWeakly(dbTrain, dbVal, ...
    'netID', 'vd16', 'layerName', 'conv5_3', 'backPropToLayer', 'conv5_1', ...
    'method', 'max', ...
    'learningRate', lr, ...
    'doDraw', true);

% --- Constructing an off-the-shelf network:

dbTrain= dbPitts(false, 'train'); % or dbTokyoTimeMachine('train');
opts.netID= 'vd16'; % or 'caffe'
opts.layerName= 'conv5_3'; % or whatever layer you want, for caffe: 'conv5'
opts.method= 'vlad_preL2_intra'; % or 'max', 'avg'
net= loadNet(opts.netID, opts.layerName); % load the network and crop at desired layer
net= addLayers(net, opts, dbTrain); % add the NetVLAD/Max/Avg layer
net= addPCA(net, dbTrain, 'doWhite', true, 'pcaDim', 4096, 'batchSize', 10); % add whitening (for non-vlad methods pick a smaller pcaDim value)



% ---------- Tiny dummy training example

% This is just to see if you got all the dependencies and configurations set up correctly.
% Get a tiny version of the Tokyo Time Machine dataset from our research page ( www.di.ens.fr/willow/research/netvlad/ ) as well as the dataset specification. Point paths.dsetRootTokyoTM in localPaths.m to its location. Run the code below to train max pooling on top of AlexNet for this tiny dataset:

dbTrain= dbTiny('train'); dbVal= dbTiny('val');

trainWeakly(dbTrain, dbVal, ...
    'netID', 'caffe', 'layerName', 'conv5', ...
    'method', 'max', 'backPropToLayer', 'conv5', ...
    'margin', 0.1, ...
    'batchSize', 4, 'learningRate', 0.01, 'lrDownFreq', 3, 'momentum', 0.9, 'weightDecay', 0.1, 'compFeatsFrequency', 10, ...
    'nNegChoice', 30, 'nNegCap', 10, 'nNegCache', 10, ...
    'nEpoch', 10, ...
    'epochTestFrequency', 1, 'test0', true, ...
    'nTestSample', inf, 'nTestRankSample', 40, ...
    'saveFrequency', 15, 'doDraw', true, ...
    'useGPU', true, 'numThreads', 12, ...
    'info', 'tiny test');

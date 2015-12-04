%  Author: Relja Arandjelovic (relja@relja.info)

% This file contains a few examples of how to train and test CNNs for place recognition, refer to README.md for setup instructions, and to our project page for all relevant information (e.g. our paper): http://www.di.ens.fr/willow/research/netvlad/


%  The code samples use the GPU by default, if you want to use the CPU instead (very slow especially for training!), add `'useGPU', true` to the affected function calls (trainWeakly, addPCA, serialAllFeats)

% For a tiny example of running the training on a small dataset, which takes only a few minutes to run, refer to the end of this file.



% Set the MATLAB paths
setup;



% ---------- Use/test our networks

% Load our network
netID= 'vd16_tokyoTM_conv5_3_vlad_preL2_intra_white';
paths= localPaths();
load( sprintf('%s%s.mat', paths.ourCNNs, netID), 'net' );

% Compute the image representation(s) by simply running the forward pass
% using the network `net` on the appropriately normalized input image(s)
% (see `serialAllFeats.m`). We also provide a utility function which does
% it all for you:
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
dbFeatFn= sprintf('%s%s_ep%06d_%s_db.bin', paths.outPrefix, bestNet.sessionID, bestNet.bestEpoch, dbTest.name);
qFeatFn = sprintf('%s%s_ep%06d_%s_q.bin', paths.outPrefix, bestNet.sessionID, bestNet.bestEpoch, dbTest.name);

% Compute db/query image representations
serialAllFeats(finalNet, dbTest.dbPath, dbTest.dbImageFns, dbFeatFn, 'batchSize', 10); % adjust batchSize depending on your GPU / network size
serialAllFeats(finalNet, dbTest.qPath, dbTest.qImageFns, qFeatFn, 'batchSize', 1); % Tokyo 24/7 query images have different resolutions so batchSize is constrained to 1

% Measure recall@N
[recall, ~, ~, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);
plot(opts.recallNs, recall, 'ro-'); grid on; xlabel('N'); ylabel('Recall@N');



% ---------- Pittsburgh examples

% Pitts250k or Pitts30k?
doPitts250k= false;

if doPitts250k
    % Pittsburgh 250k
    lr= 0.001;
else
    % Pittsburgh 30k
    lr= 0.0001;
end

dbTrain= dbPitts(doPitts250k, 'train');
dbVal= dbPitts(doPitts250k, 'val');
dbTest= dbPitts(doPitts250k, 'test');

% Now just run the same code as above for Tokyo



% ---------- Miscellaneous
% Other potentially useful

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

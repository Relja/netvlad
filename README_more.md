# NetVLAD: CNN architecture for weakly supervised place recognition

This is an additional collection of comments regarding the usage of the NetVLAD code, please refer to `README.md` first. More comments are available in the code itself.

For the link to the paper, trained models and other data, see our project page:
http://www.di.ens.fr/willow/research/netvlad/

#### Training parameters

Here is a brief description of some parameters that can be passed to `trainWeakly`. Their default values are available at the beginning of `trainWeakly.m`.

Main parameters:

 - `netID`: The name of the network (`caffe` for AlexNet, `vd16` for verydeep-16, i.e. VGG-16)
 - `layerName`: Which layer to crop the initial network at, we always use the last convolutional layer (i.e. conv5 for caffe and conv5_3 for vd16)
 - `method`: Which aggregation method to use for the image representation, default is `vlad_preL2_intra` (i.e. NetVLAD with input features L2-normalized, and with intra-normalization of the NetVLAD vector). You can also use `max` for max pooling, `avg` for average pooling, or other vlad variants (e.g. `vlad_preL2` to disable intra-normalization)
 - `margin`: The margin parameter for the loss
 - `useGPU`: Use the GPU or not
 - `sessionID`: A string used to identify the run, if empty a random one will be supplied for you. Careful, it is used in naming files so probably don't put slashes or other non-alphanumeric characters.
 - `computeBatchSize`: The batch size used to compute image representations (**not** the same as SGD training batch size) - changing this parameters does not change the training behaviour, just the speed of computing the cached representations (bigger is generally better, but it depends how much you can fit on your GPU).
 - `compFeatsFrequency`: Recompute the cached image representation every `compFeatsFrequency` training tuples (queries).

SGD parameters:

 - `learningRate`, `batchSize`, `momentum`, `weightDecay`, `nEpoch`: SGD parameters with obvious meanings. Note: `batchSize` refers to the number of training tuples in a batch, but tuples contain many images (query, best potential positive, negatives).
 - `lrDownFactor`, `lrDownFreq`: The learning rate is decreased by a factor of `lrDownFactor` every `lrDownFreq` epochs, and the `compFeatsFrequency` is multiplied by the same factor at the same intervals, as cache becomes stale slower with smaller learning rates.
 - `backPropToLayer`: Down to which layer to perform the learning. If not specified, the entire network is trained. You can use the name of the layer, or the layer number.
 - `fixLayers`: Cell array of layers which should not be changed at training - avoid this and use `backPropToLayer` instead. Only useful if you want to have some exotic experiments (e.g. fix the NetVLAD layer but update the lower layers - probably a strange thing to do but it can be done).

Other parameters:

 - `doDraw`: To plot or not some performance curves as training goes along.
 - `printLoss`, `printBatchLoss`: print all loses as training goes along - only use for debugging as your screen will be cluttered.
 - `saveFrequency`: Save the network and other data every `saveFrequency` training queries (+ saved once at the end of each epoch)
 - `recallNs`: Measure recall at what N values (array)? Easier to manage if you stick to one setting for all your training runs (e.g. the default)
 - `numThreads`: Number of threads passed to vl_jpegread (i.e. image loading threads)
 - `test0`: Test the network before training (i.e. off-the-shelf mode) or not
 - `epochTestFrequency`: Every how many epochs do you want to test the network on train and val (probably best to keep it =1)
 - `nTestRankSample`: Number of tuples used to compute the loss when the network is tested; same tuples are used across epochs/training runs, provided the value of the parameter is the same.
 - `nTestSample`: Upper limit on the number of queries used to test the network; same queries are used across epochs/training runs, provided the value of the parameter is the same. Set to $\infty$ if you want to use them all.
 - `nNegChoice`: Number of negatives to sample randomly for every training tuple (for each epoch independently)
 - `nNegCache`: Number of hardest negatives to remember for the following epoch
 - `nNegCap`: Number of hardest negatives out of the `nNegChoice`+`nNegCache` that are kept in the training tuple.
 - `excludeVeryHard`: If set to true, it ignores negatives which are closer than the closest potential positive, thus only keeping "semi-hard" negatives (i.e. ones which violate the margin but not so much). I didn't play with this much and in all our experiments this parameter was set to false. I don't think it was beneficial when I briefly tried setting it to true, but the [FaceNet](http://arxiv.org/abs/1503.03832) paper which introduced the idea finds it to be crucial. Maybe it will be important if you try to train the network from scratch.
 - `startEpoch`: Allows you to restart the training from a given epoch (you should provide the appropriate `sessionID`). You can change the learning rate or other SGD parameters if you want. Take care because 1) it is not well tested but should work, 2) the provided learning rate is used as the rate at epoch 1 and not as the rate at the starting epoch (i.e. it will potentially be decreased according to the provided learning schedule, see the code).
 - `dbCheckpoint0`, `qCheckpoint0`, `dbCheckpoint0val`, `qCheckpoint0val`: Where to save the initial image representation files for the database(db)/query(q) for the train/val sets using the off-the-shelf network. These are needed to test the network (if `test0` is true) and are naturally used as the initial image representation cache. Best to leave them empty as the filenames will managed automatically.
 - `checkpoint0suffix`: You can specify a short string which will be appended to the automatically generated filenames above (`dbCheckpoint0`, `qCheckpoint0`, `dbCheckpoint0val`, `qCheckpoint0val`), if those filenames were left empty.

It is important not to run training with the same combination of [ training dataset (`dbTrain`), `netID`, `layerName` and `method` ], for example, if you wish to try different learning rates at the same time. This is because the initial image representations computed using the off-the-shelf network (see `dbCheckpoint0`, ..) are going to be stored in the same file, so multiple processes will be simultaneously writing to the same file, corrupting it, or some of them will fail (they will detect the files already exist, will try to load them but they are corrupt so it will fail). If you do want to run multiple trainings with the same combination of the aforementioned settings, your options are:

1. Wait for one process to compute the 4 files`dbCheckpoint0`, `qCheckpoint0`, `dbCheckpoint0val`, `qCheckpoint0val` and only then run the other ones.
2. Set different `dbCheckpoint0`, `qCheckpoint0`, `dbCheckpoint0val`, `qCheckpoint0val` for different runs manually (repeating the computations unnecessarily).
3. Like (2). but more convenient - just set a different `checkpoint0suffix` for each run as the 4 generated file names will differ.

#### Training output format, performance curves

The training periodically saves files which contain the current network (see `README.md`). Apart from the network (`net`), it also saves the options passed to `trainWeakly` (`opts`), a structure containing various performance curves (e.g. validation recall@N) in `obj`, and some more data in `auxData` (e.g. hard negatives). To load these do:

    load('0fd5_ep000020_latest.mat', 'obj', 'opts', 'auxData');

Then you can plot the same curves as `trainWeakly` plots if `doDraw` is set to true:

    plotResults(obj, opts, auxData);

The top left curve shows the current weakly supervised triplet ranking loss (smoothed) on the training batches (I call it "dynamic loss" here). This is the only curve shown if you're still in the first epoch.

The bottom left curve shows the same loss but evaluated on train and validation sets, using random samples of tuples (see the `nTestRankSample`).

The right curve is the most important one - it shows recall@N for train and validation sets, where the legend c.N signifies: c=t for train, c=v for test, N= number of top images used for recall@N. So recall@5 on the validation set is marked as v.5 on this curve. Recall from earlier that only up to `nTestSample` (=1000 by default) are used for these tests.

#### Training observations and tips

The "dynamic loss" (see above) can be a bit deceiving - due to hard negative mining it might not go down for a while, even though the training is working correctly. E.g. it is not uncommon for it to be at roughly the same level, or even increase a bit, for a few epochs, even though the thing we care about (recall@N) improves. This makes it quite hard to do efficient search for optimal SGD parameters as it is not obvious if SGD is working (unless the error increases widely in which case it's obviously not).

If your "dynamic loss" is exhibiting some periodic behaviour (e.g. very pronounced bumps at relatively regular intervals), it's is likely that the cached representations (see implementation details of our paper) are not refreshed often enough for your data. In this case the network overfits to a cache which causes the "dynamic loss" to go down for a while, and when the cache is refreshed it suddenly jumps (and due to smoothing for display this looks more like a smooth bump than a sharp rise). To fix, decrease the `compFeatsFrequency` parameter. On the other hand, if you have no such problems and training is slow - you can speed it up by increasing `compFeatsFrequency`, as the cache computation is often the bottleneck.

Relatively low hanging fruits:

- Optimize SGD parameters: I didn't have the time / computational resources to do this thoroughly and it's likely you can find better combinations of margin, learning rate, momentum, weight decay, learning rate schedule, cache recomputation frequency, batch size, number of hard negatives, etc. Maybe also train the network multiple times using different random seeds and pick the best.
- Jittering: Again, I didn't have the time for it, but it should really help counter the overfitting. However, one should think carefully if different scales should be used as scale invariance might not be beneficial for place recognition (if your image contains the Eiffel tower from 1km away, guessing you're next to it is bad for place recognition). I might include some jittering in potential further versions of the code.
- Try cropping the input networks at different layers: we always used the last convolutional layer as input to NetVLAD, but it could be the case that lower levels are more appropriate. Set `layerName` in `trainWeakly` to do this.
- Maybe NetVLAD training can be faster if pretraining is done with max pooling. Then perform further tuning by cropping off the max pooling and putting the NetVLAD layer on top. I haven't tried it, but I imagine it would work.

#### Output NetVLAD vector details

Here we give details of the organization of the output NetVLAD layer. Note that we recommend to perform whitening on top (and networks with whitening incorporated are provided) as the place recognition results are superior, and the memory footprint is smaller. The discussion below is purely related to the direct output of the NetVLAD layer (i.e. the VLAD-like vector), and not the final output of our best networks (i.e. smaller whitened vector).

The NetVLAD layer produces an output which is (KxD)x1 dimensional (K=number of cluster centres, D= dimensionality of the input descriptors). The vector is obtained by doing `reshape(netvladMatrix, [K*D, 1])` where `netvladMatrix` is KxD, so, for example, values corresponding to the first cluster centre are stored in [1:K:end] locations. This is important to keep in mind if you're interested in compressing the vector using techniques such as [Product Quantization](http://people.rennes.inria.fr/Herve.Jegou/projects/ann.html) because it is likely that you want to make the splitting into subquantizers according to cluster centres for the PQ to work better. So, it is likely that it would be beneficial to do:

    netvladForPQ= reshape( reshape(netvlad, [K, D])', [K*D, 1] );

as this will group values associated to cluster centres in blocks of size D (e.g. 1st cluster centre: [1:D], 2nd: D+[1:D], etc).

#### Output binary files

The `serialAllFeats` function, which computes representations for sets of given images, stores these files by simply writing the values in single 32-bit floats, image by image. So, for a D-dimensional image representation, the output file will contain: D values for 1st image, D values for the 2nd image, etc. The size of the output file is `D x numImages x 4` bytes. To read the files in matlab, do:

    feats= fread( fopen(featFn, 'rb'), [D, numImages], 'float32=>single');

#### Tokyo 24/7 detailed tests

Tokyo 24/7 defines daytime / sunset / nighttime queries. The example from README.md shows how to test using all queries together. To get more detailed results, add an extra output variable like this:

    [recall, ~, allRecalls, opts]= testFromFn(dbTest, dbFeatFn, qFeatFn);

Now `allRecalls` contains recall@N (where the order of Ns is according to `opts.recallNs`) for all queries. To get the recall for all queries, do

    recalls= mean( allRecalls, 1 )';

The queries in Tokyo 24/7 are organized such that (counting from 1) `iQuery mod 3` gives 1-daytime, 2-sunset, 0-nighttime. So to get, for example, only performance on daytime queries, do:

    recallDay= mean( allRecalls(1:3:end,:), 1)';

Similarly, to get recall for sunset and nighttime queries together:

    recallSunsetNight= mean( allRecalls([2:3:end, 3:3:end],:), 1)';

#### NetVLAD versions

There are two versions of the NetVLAD layer in the code: `layerVLAD` and `layerVLADv2` (they are very similar - do the `diff`). `layerVLADv2` follows the paper exactly, while `layerVLAD` is a version where we fix $b_k$ parameters (i.e. they are not updated during training). Assuming that input descriptors are L2-normalized (**don't use `layerVLAD`otherwise**!!), which seems to anyway work better with VLAD and NetVLAD when combined with conv5 descriptors, the assignment into clusters can be done more simply by scalar product (i.e. initially $w_k$=L2-normalized  cluster centre, $b_k$=0). This version seems to converge faster, so it is the version we use in all our experiments. It might well be that `layerVLADv2` can get you better results - it should work at least as well as `layerVLAD`, unless you are overfitting too much, as it has the capacity to do everything `layerVLAD` can do, and more!

#### Off-the-shelf networks

You can download off-the-shelf networks from our [project page](http://www.di.ens.fr/willow/research/netvlad/), or you can construct them yourselves by 1) downloading the original off-the-shelf networks from the [website](http://www.vlfeat.org/matconvnet/), and 2) adding the required pooling layers.
The second can be done using the `addLayers` function - see how it is used in `trainWeakly`. For NetVLAD one needs to also provide cluster centres (which depend on the network + layers + training dataset) - the code automatically loads them if you have them precomputed (you can download ours from the project page), or it computes them itself if the files don't exist.

% Training File 
% Brian Zenger
% CS 6640 
% Project 5

vl_compilenn('enableGpu', true, 'cudaRoot', '/usr/local/cuda-8.0/','enableImreadJpeg',true)
vl_setupnn

%% Creating the Networks for Even Odd calculation
figure(1)
[net,info] = cnn_mnist;

rng('default');
rng(0) ;
f=1/100;
netEO = net;
EXTRA_LAYER = 1; % put a one if you want to add an extra layer to the network, put a zero if you want to edit the last layer of the network
switch EXTRA_LAYER
    case 1
        netEO.layers{9} = netEO.layers{8};
        netEO.layers{8} = struct('type', 'conv', ...
            'weights', {{f*randn(1,1,10,2, 'single'), zeros(1,2,'single')}}, ...
            'stride', 1, ...
            'pad', 0) ;
        for ii = 1:7
            netEO.layers{ii}.learningRate=[0,0]; % Freeze the weights learned from before
        end
    case 0
        netEO.layers{7}.weights={f*randn(1,1,500,2, 'single'), zeros(1,2,'single')}; %Reset the Final classification and have two output nodes
        for ii = 1:6
            netEO.layers{ii}.learningRate=[0,0]; % Freeze the weights learned from before
        end
end
netEO.meta.trainOpts.learningRate=1e-4;
netEO.meta.classes.name = {'0','1'}; % Can change this to {'even','odd'} ???? - gets overwritten inside cnn_mnist anyway
netEO = vl_simplenn_tidy(netEO);
expDir = fullfile(vl_rootnn, 'matconvnet-1.0-beta25/data', ['mnist-evenodd-' 'simplenn']);

%load the image database, edit the labels to be even/odd, then save so it
%can be loaded by the folloiwng function
imdb = load('matconvnet-1.0-beta25/data/mnist-baseline-simplenn/imdb.mat');
imdb.images.labels=mod(imdb.images.labels,2); %0 is even, 1 is odd

if ~exist(expDir, 'dir'), mkdir(expDir) ; end
save('matconvnet-1.0-beta25/data/mnist-evenodd-simplenn/imdb.mat','-struct','imdb','images','meta');
figure(2)
netEO = cnn_mnist('network',netEO,'expDir',expDir,'imdbPath','data/mnist-evenodd-simplenn/imdb.mat');
netEO.meta.classes.name = {'Even','Odd'};
%% CIFAR Dog/Cat Classifier Test
% modify data
cifar = load('matconvnet-1.0-beta25/data/cifar-lenet/imdb.mat');
catdog = struct();
catdog.images.data = cifar.images.data(:,:,:,cifar.images.labels == 4 | cifar.images.labels == 6);
catdog.images.labels = 1 + (cifar.images.labels(cifar.images.labels == 4 | cifar.images.labels == 6) == 6);
catdog.images.set = ones(size(catdog.images.labels)); % Just use everything for training.  Not the best choice in the world
catdog.meta.classes={'cat','dog'};

%% modify network
rng('default');
rng(0) ;
f=1/100;
netC = netEO;
netC.meta.trainOpts.batchSize = 1024;
netC.meta.trainOpts.learningRate=5e-4;
netC.meta.trainOpts.numEpochs=80;
netC.layers{end}.type = 'softmaxloss'; %training doesnt work unless this, but needs 'softmax' for inference??? Weird Deprecations Bro
for ii = 1:7
    % Make whole network trainable
    netC.layers{ii}.learningRate=[1,1];
    if netC.layers{ii}.type == 'conv'
        % 'Reset' conv weights while preserving sizes
        netC.layers{ii}.weights = { f * randn(size(netC.layers{ii}.weights{1}),'single'), randn(size(netC.layers{ii}.weights{2}),'single')};
    end
end
netC.meta.inputSize = [32 32 3] ;
netC.layers{1} = struct('type', 'conv', ... #        v this changed from 1 to 3 because rgb for immediate reduction in first layer
    'weights', {{f*randn(6,6,3,20, 'single'), zeros(1, 20, 'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
netC = vl_simplenn_tidy(netC);
vl_simplenn_display(netC, 'batchSize', netC.meta.trainOpts.batchSize) ;


%% Train
outDir = 'matconvnet-1.0-beta25/catdog/';
if ~exist(outDir, 'dir'), mkdir(outDir) ; end
opts.train = struct() ;
opts.train.gpus = [2,3,4,5];
opts.train.momentum = 0.8; %larger batch sizes make this okayish?
opts.networkType = 'simplenn';
[netCtrained, infoCtrained] = cnn_train(netC, catdog, getBatch(opts), ...
    'expDir', outDir, ...
    netC.meta.trainOpts, ...
    opts.train) ;

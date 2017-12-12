% Deep Learning Assignment Project 5
% Brian Zenger, U0291777
% Bioen 6640 Image Processing

%% run this during the matlab startup if networks already created

tempnet = load('matconvnet-1.0-beta25/data/mnist-baseline-simplenn/net-epoch-20.mat')
net = tempnet.net
tempnet = load('matconvnet-1.0-beta25/catdog/net-epoch-80.mat')
netEO = tempnet.net
tempnet = load('matconvnet-1.0-beta25/data/mnist-evenodd-simplenn/net-epoch-20.mat')
netC = tempnet.net
%% Viewing the weights of the first layer
%montage(net.layers{1}.weights{1})

figure(1)
for k = 1:20
subplot(4,5,k)
imagesc(net.layers{1}.weights{1}(:,:,1,k));axis image; colormap gray; 
title(['Filter ' num2str(k)])
end
print('OutputImages/Weights','-depsc');
% Finish this part??

%% Number clasify net Manual Spot Test
net.layers{end}.type = 'softmax';
imdbNum = load('matconvnet-1.0-beta25/data/mnist-baseline-simplenn/imdb.mat');

for ii = [1,2,3] %image numbers to be tested. 
    testim = imdbNum.images.data(:,:,1,ii);
    testim = single(testim);
    testim = imresize(testim, net.meta.inputSize(1:2));
    
    %run it
    res = vl_simplenn(net,testim);
    
    %Classify Result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    class= net.meta.classes.name{best};
    numberName=class;
    number = num2str(ii);
    figure(4); clf; imagesc(testim); axis image; colormap gray; drawnow;
    title(sprintf('the number is %s, score is %.1f%%',numberName,bestScore*100))
    print(['OutputImages/MINST' number],'-depsc');
    
    
end


%% Even odd net Manual Spot Test
netEO.layers{end}.type = 'softmax';
imdbEO = load('matconvnet-1.0-beta25/data/mnist-evenodd-simplenn/imdb.mat');

for ii = [1,6,8]
    testim = imdbEO.images.data(:,:,1,ii);
    testim = single(testim);
    testim = imresize(testim, net.meta.inputSize(1:2));
    %run it
    res = vl_simplenn(netEO,testim);
    
    %Classify Result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores); % This is the correct index into the class names cell array on the line below (by itself this number is confusing)
    class= netEO.meta.classes.name{best};
    numberName=class;
    number = num2str(ii);
    figure(3); clf; imagesc(testim); axis image; colormap gray; drawnow;
    title(sprintf('the number is %s, score is %.1f%%',numberName,bestScore*100))
    print(['OutputImages/EvenOdd' number],'-depsc');
    
end

%% Preprocessing images to be fed into the system

testImage = imread('InputImages/Handwritetest.png');
if size(testImage,3) == 3
    testImage = rgb2gray(testImage);
end

testImage = testImage/max(max(testImage));
imageThresholded = testImage < 0.1;
labelImage = zeros(size(testImage,1),size(testImage,2));
[CCImage, endingLabelValue] = connectedComponentBZ(imageThresholded,1,labelImage);
targetSize = [28,28];
for ii=1:endingLabelValue
    threshold_image = CCImage ==ii;
    [x,y]=find(CCImage==ii);
    EDGE_BIAS = 20;
    Lboundx = min(x)-EDGE_BIAS;
    if Lboundx <=0
        Lboundx =1;
    end
    Lboundy = min(y)-EDGE_BIAS;
    if Lboundy <=0
        Lboundy =1;
    end
    Hboundx = max(x)+EDGE_BIAS;
    if Hboundx > size(CCImage,1)
        Hboundx = size(CCImage,1);
    end
    Hboundy = max(y)+EDGE_BIAS;
    if Hboundy > size(CCImage,2)
        Hboundy = size(CCImage,2);
    end
    temp_var = strcat( 'image_',num2str(ii));
    temp_image = double(threshold_image(Lboundx:Hboundx,Lboundy:Hboundy));
    sourceSize = size(temp_image);
    [X_samples,Y_samples] = meshgrid(linspace(1,sourceSize(2),targetSize(2)), linspace(1,sourceSize(1),targetSize(1)));
    temp_image = interp2(temp_image, X_samples, Y_samples);
    temp_image = temp_image > 0.75;
    temp_image = 255*single(temp_image)-imdb.images.data_mean;
    res = vl_simplenn(net,temp_image);
    
    %Classify Result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    best = best-1;
    if best ==1
        numberName = 'odd';
    else
        numberName = 'even';
    end
    number = num2str(ii);
    figure(ii); clf; imagesc(temp_image); axis image; colormap gray; drawnow;
    title(sprintf('the number is %s, score is %.1f%%',net.meta.classes.name{best+1}-1,bestScore*100))
    print(['OutputImages/HandWrittenImage' number],'-depsc');
    temp_var_im = 'temp_image';
    images{ii} = temp_image;
end



%% CIFAR dataset
figure()
for ii = 1,2,3
    netC.layers{end}.type = 'softmax';
    testim = catdog.images.data(:,:,:,ii);
    testim = single(testim);
    testim = reshape(testim, netC.meta.inputSize);
    
    %run it
    res = vl_simplenn(netC,testim);
    
    %Classify Result
    scores = squeeze(gather(res(end).x));
    [bestScore, best] = max(scores);
    class= catdog.meta.classes{best};
    animalName=class;
    animal = num2str(ii);
    figure(ii); clf; imagesc(testim); axis image; colormap gray; drawnow;
    title(sprintf('the animal is %s, score is %.1f%%',animalName,bestScore*100))
    print(['OutputImages/CIFAR' animal],'-depsc');
end


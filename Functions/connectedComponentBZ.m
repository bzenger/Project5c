function [labelImage,endingLabelValue]=connectedComponentBZ(binaryImage,startingLabelValue,labelImage)

binaryImage = logical(binaryImage);
labelImage(binaryImage)=-1; %matlab magic

for count=1:size(labelImage,1)
    for count1=1:size(labelImage,2)
        if labelImage(count,count1)==-1
            labelImage=floodfill(binaryImage,count,count1,startingLabelValue,labelImage);
            startingLabelValue=startingLabelValue+1;
        end
       
    end
end

endingLabelValue=startingLabelValue-1;

end


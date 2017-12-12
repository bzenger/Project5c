%floodfill function

function labelImage = floodfill(inputBinaryImage, xs, ys, labelValue, labelImage); 

% [xlength,ylength]= size(inputImage);
% outputImageMask=zeros(xlength,ylength);

spread=[];
visited=zeros(size(inputBinaryImage));


if inputBinaryImage(xs,ys)== 1
    spread=[xs,ys];
end

while ~isempty(spread)
    % Pick a pixel 
    currentX= spread(1,1);
    currentY= spread(1,2);
    spread(1,:)=[];
    if visited(currentX,currentY)
        continue
    end
    % add that pixel to visited list 
    visited(currentX,currentY)=visited(currentX,currentY)+1;
    % conditional that pixel ==1 
    if inputBinaryImage(currentX,currentY)==1 
        % assign new object mask = 1
        labelImage(currentX,currentY)=labelValue;
        % add current four neighbors to spread list if they have not been visited
            if currentX+1<=size(labelImage,1)
            spread(end+1,:)=[currentX+1,currentY];
            end
            if currentX-1>0
            spread(end+1,:)=[currentX-1,currentY];
            end 
            if currentY+1<=size(labelImage,2)
            spread(end+1,:)=[currentX,currentY+1];
            end
            if currentY-1>0
            spread(end+1,:)=[currentX,currentY-1];
            end
            
    end

end



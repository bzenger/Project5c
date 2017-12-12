function labelImage= topologicalDenoising(endingLabelValue, labelImage, smallestSize)

voteMatrix=zeros(endingLabelValue);

for count=1:size(labelImage,1)
    for count1=1:size(labelImage,2)
        if count+1<=size(labelImage,1)
            voteMatrix(labelImage(count,count1),labelImage(count+1,count1))=voteMatrix(labelImage(count,count1),labelImage(count+1,count1))+1;
        end
        if count-1>0
            voteMatrix(labelImage(count,count1),labelImage(count-1,count1))=voteMatrix(labelImage(count,count1),labelImage(count-1,count1))+1;
        end
        if count1+1<=size(labelImage,2)
            voteMatrix(labelImage(count,count1),labelImage(count,count1+1))=voteMatrix(labelImage(count,count1),labelImage(count,count1+1))+1;
        end
        if count1-1>0    
            voteMatrix(labelImage(count,count1),labelImage(count,count1-1))=voteMatrix(labelImage(count,count1),labelImage(count,count1-1))+1;
        end
    end 
end

voteMatrix(logical(eye(size(voteMatrix))))=0;

for ii=1:size(labelImage,1)
    for jj=1:size(labelImage,2)
        if sum(labelImage(:)==labelImage(ii,jj))< smallestSize
            [~,replacementLabel]=max(voteMatrix(labelImage(ii,jj),:));
            labelImage(ii,jj)= replacementLabel;
        end
    ii    
    end
end

end

function descriptor = histDesc(image1, features)
% Descriptor Extraction
% Design 1: 11x11 Sampled Patches

% image1_padded = padarray(image1,[5 5]);
desc_num = size(features,2);
patch = zeros(desc_num,32*32);
descriptor = zeros(desc_num,121);
[imag_height, imag_width] = size(image1);
for desc_idx = 1:desc_num
    
    idx = 1;
    for x = min(max(features(1,desc_idx)-16,1),imag_height-31) : min(max(features(1,desc_idx)+15,32),imag_height)
        for y = min(max(features(2,desc_idx)-16,1),imag_width-31) : min(max(features(2,desc_idx)+15,32),imag_width)
            patch(desc_idx,idx) = image1(x,y);
            idx = idx + 1;
        end
    end
    
%     descriptor(desc_idx,:) = (descriptor(desc_idx,:)-mean(descriptor(desc_idx,:)))./std(descriptor(desc_idx,:));
    descriptor(desc_idx,:) = histcounts(patch(desc_idx,:),121);
end

end
function descriptor = sampledPatchDesc(image1, features)
% Descriptor Extraction
% Design 1: 11x11 Sampled Patches

% image1_padded = padarray(image1,[5 5]);
desc_num = size(features,2);
descriptor = zeros(desc_num,11*11);
[imag_height, imag_width] = size(image1);
for desc_idx = 1:desc_num
    
    idx = 1;
    for x = min(max(features(1,desc_idx)-5,1),imag_height-10) : min(max(features(1,desc_idx)+5,11),imag_height)
        for y = min(max(features(2,desc_idx)-5,1),imag_width-10) : min(max(features(2,desc_idx)+5,11),imag_width)
            descriptor(desc_idx,idx) = image1(x,y);
            idx = idx + 1;
        end
    end
    
    descriptor(desc_idx,:) = (descriptor(desc_idx,:)-mean(descriptor(desc_idx,:)))./std(descriptor(desc_idx,:));
end

end
%% Generate training set
function patches = loadSampleData()
%% To get a single training example x
% randomly pick one image and randomly sample
% image patches
% Returns concatenated matrix of patches
load /../../../../data/Image/IMAGES;

patchSize = 8; % Use 8x8 patches
numpatches = 10000;

for i = 1 : numpatches 
    % IMAGES is a 3D array containing 10 images
    % Randomly pick one of the images
    image_index = randi([1 10],1,1);
    % Randomly sample an 8x8 image patch
    patch = zeros(patchSize,patchSize);
    rand_point = randi([1 (512 - patchSize)],1,1);
    patch = IMAGES(rand_point:rand_point+patchSize-1, ...
        rand_point:rand_point+patchSize-1,image_index);
    % Convert the image patch to a single column vector
    patch_col = patch(:);
    if (i == 1)
        patches = [patch_col];
    else
        patches = [patches(:,:) patch_col];
    end
end


%% ---------------------------------------------------------------
% Normalize the data
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;
end
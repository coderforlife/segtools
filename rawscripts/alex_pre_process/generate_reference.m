function generate_reference( path_in, path_out, N )
% Generate a reference histogram for exact histogram specification
%    Generates a histogram with 256 bins for an input 8-bit image. Prior
%    to histogram calculation, borders are removed from the input image.
%    The output histogram is stored to an ASCII text file.
%    
%    INPUT
%    ----------
%    path_in     Path to the stack of TIF/PNG files to process
%    path_out    Path to output the histogram text file to
%    N           Image number to analyze within path_in
%

%Parse path_in for test images
imgs = dir(fullfile(path_in,'*.png'));
if isempty(imgs)
    imgs = dir(fullfile(path_in,'*.tif'));
end

file_out = fullfile(path_out,['ref_hist_' sprintf('%04d',N) '.txt']);

%Compute overall histogram. Each image is read, and its borders are removed.
%The image histogram is computed only from the pixels remaining following
%border removal.
file_in = fullfile(path_in,imgs(N).name);
img_in = uint8(imread(file_in));
[IR IC] = size(img_in);
[PV_img,~] = find_nonborder_pixels(img_in);
clear img_in
fprintf('Analyzing %s\n',file_in);
fprintf('Image size = %d\n',IR*IC);
fprintf('Border size = %d\n',IR*IC - numel(PV_img));
hist_img = hist(double(PV_img),256);
hist_img = hist_img';

save(file_out,'hist_img','-ascii');
fprintf('Output written to %s\n',file_out);

end

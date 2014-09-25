function run_ehs( path_imgs, path_ref, path_out, N )
% 
%
%
%
%
%
%
%

files_ref = dir(fullfile(path_ref,'*.txt'));
imgs = dir(fullfile(path_imgs,'*.png'));
if isempty(imgs)
    imgs = dir(fullfile(path_imgs,'*.tif'));
end

%Compute summed histogram of all references
hist_sum = zeros(256,1);
for i = 1:numel(files_ref)
    hist_i = load(fullfile(path_ref,files_ref(i).name));
    hist_sum = hist_sum + hist_i;
end

I = uint8(imread(fullfile(path_imgs,imgs(N).name)));
[PV_init,border] = find_nonborder_pixels(I);

[ehs,~] = exact_histogram(I,hist_sum,border);
clear I border

[PV_ehs,~] = find_nonborder_pixels(ehs);

for i = 1:256
    fprintf('%d %d %d %d\n',i-1,hist_sum(i),PV_init(i),PV_ehs(i));
end

%Write output
[~,base,ext] = fileparts(imgs(N).name);
file_out = fullfile(path_out,[base '_EHS' ext]);
imwrite(uint8(ehs),file_out);
fprintf('Output written to %s\n',file_out);

end


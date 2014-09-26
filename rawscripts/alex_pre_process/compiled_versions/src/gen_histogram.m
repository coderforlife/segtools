function gen_histogram( input_img_file, output_txt_file)
% Generate a reference histogram for exact histogram specification
%    Generates a histogram with 256 bins for an input 8-bit image. Prior
%    to histogram calculation, borders are removed from the input image.
%    The output histogram is stored to an ASCII text file.
%    
%    INPUT
%    ----------
%    input_img_file     Image (TIF/PNG) file to process
%    output_txt_file    Path to output the histogram text file
%

  %Compute overall histogram. Each image is read, and its borders are removed.
  %The image histogram is computed only from the pixels remaining following
  %border removal.
  img_in = uint8(imread(input_img_file));
  [IR IC] = size(img_in);
  [PV_img,~] = find_nonborder_pixels(img_in);
  clear img_in
  fprintf('Analyzing %s\n',input_img_file);
  fprintf('Image size = %d\n',IR*IC);
  fprintf('Border size = %d\n',IR*IC - numel(PV_img));
  hist_img = hist(double(PV_img),256);
  hist_img = hist_img';

  save(output_txt_file,'hist_img','-ascii');
  fprintf('Output written to %s\n',output_txt_file);

end

function run_ehs( input_image, histogram_file , output_image )
% Generates histogram equalized image from input image and histogram
% text file.
%
%   INPUT
%   ----------
%   input_image      Input image to process
%   histogram_file   256 bin histogram text file to use for histogram
%                    equalization
%   output_image     Path to write histogram equalized image
%


  %Compute summed histogram of all references
  hist_sum = zeros(256,1);
  hist_sum = load(histogram_file);

  I = uint8(imread(input_image));
  [PV_init,border] = find_nonborder_pixels(I);

  [ehs,~] = exact_histogram(I,hist_sum,border);
  clear I border

  [PV_ehs,~] = find_nonborder_pixels(ehs);

  for i = 1:256
      fprintf('%d %d %d %d\n',i-1,hist_sum(i),PV_init(i),PV_ehs(i));
  end

  %Write output
  imwrite(uint8(ehs),output_image);
  fprintf('Output written to %s\n',output_image);
end


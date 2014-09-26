function [PV,border] = find_nonborder_pixels( I )
% Returns the pixel values of an input image that are not part
% of the border. The border is determined by computing the
% gradient magnitude of the image, then searching for pixels
% with gradient values that are approximately zero. This method
% of determining the image border works for all types of borders,
% including simple linear translations and shears.
%
%    INPUT
%    ----------
%    I        Image to extract pixel values from
%
%    OUTPUT
%    ----------
%    PV       1xM vector of non-border pixel values
%    border   Binary image displaying the border pixels
%

[FX,FY] = gradient(double(I));
border = sqrt(FX.^2 + FY.^2); %Gradient magnitude
border = (border < 0.01); %Find where gradient is ~zero
border = imfill(~border,'holes');
clear FX FY

RP = regionprops(border,I,'PixelValues');
PV = []; %Initialize pixel value vector
for i = 1:numel(RP)
    PV = [PV RP(i).PixelValues'];
end

end

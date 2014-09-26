% You need to run mbuild -setup before running this script for the first time!

cd ../src

% Compile the programs
% -m creates a standalone C command line application
% -R specifies matlab runtime arguments
% -N removes all toolbox paths
% -p readds a toolbox path
% -I includes a folder while compiling (does not necessary include all files in it if they are not found needed)
% -o gives the output filename
% -d gives the output directory
% Everything at the end are the functions to include/export
fprintf('Compiling gen_histogram...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -p images/iptformats -p images/imuitools -d ../compiled gen_histogram

fprintf('\nCompiling merge_histograms...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -d ../compiled merge_histograms

fprintf('\nCompiling run_ehs...\n');
mcc -m -R '-nojvm,-nodisplay,-singleCompThread' -N -p images/images -p images/iptutils -d ../compiled run_ehs



cd ../compiled

% Remove extraneous files
delete run_gen_histogram.sh
delete run_merge_histograms.sh
delete run_run_ehs.sh
delete readme.txt
delete mccExcludedFiles.log

% Save MATLAB and compiler version which last compiled the code
fid = fopen('matlab-version.txt','w');
fprintf(fid,'Platform:         %s\n',computer);
fprintf(fid,'MATLAB Version:   %s\n',version);
[maj,min,rev] = mcrversion;
fprintf(fid,'Compiler Version: %d.%d.%d\n',maj,min,rev); 
fclose(fid);

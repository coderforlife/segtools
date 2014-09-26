% Runs MATLAB code from the command line.
% The code should be sent as a single string.
% If any exceptions are thrown by the code they are caught, displayed, and MATLAB exits with return value 1 instead of 0.
% If this script is not used and an exception happens then MATLAB corrupts the shell completely.
% Any time MATLAB is run from the shell *nix you should always run "stty sane" afterwards.
function run_from_shell(str)
    runFromShellStart=tic;
    assignin('base','result',1);
    c=onCleanup(@() evalin('base','exit(result);')); % required for CTRL+C handling
    try
        eval(str);
        assignin('base','result',0);
    catch exc
        fprintf(2, getReport(exc));
    end
    fprintf('Running << %s >> took %f seconds\n',str,toc(runFromShellStart));

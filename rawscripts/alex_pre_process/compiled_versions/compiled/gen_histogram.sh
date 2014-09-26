#!/bin/bash

usage()
{
  echo "Generate Histogram from Image Script.  @@VERSION@@

$0 <input image> <output txt file> <optional arguments>
  inputs            Input image, should be 8 bit greyscale image 
  output txt file   ASCII text file containing histogram of pixel
                    intensities.  This file has 256 lines, 1 line
                    for ever pixel intensity.

Optional Arguments:
  -M matlab_dir   MATLAB or MCR directory. If not given will look for a MCR_DIR
                  environmental variable. If that doesn't exist then an attempt
                  will be made using 'which'. It must be the same version used
                  to compile the scripts." 2>&1;
  exit 1;
}

# Parse and minimally check arguments
if [[ $# -lt 2 ]]; then usage; fi
if [[ $# -gt 2 ]] && [ "${3:0:1}" != "-" ]; then
  echo "You provided more than 2 required arguments. Did you accidently use a glob expression without escaping the asterisk?" 1>&2; echo; usage; 
fi
INPUT=$1;
OUTPUT=$2;
MATLAB_FOLDER=;
shift 2
while getopts ":M:" o; do
  case "${o}" in
    M)
      MTLB_FLDR=${OPTARG};
      if [ ! -d "$MTLB_FLDR" ]; then echo "MATLAB folder is not a directory." 1>&2; echo; usage; fi;
      ;;
    *)
      echo "Invalid argument." 1>&2; echo; 
      usage;
      ;;
    esac
done

# Find MATLAB or MATLAB Compiler Runtime and add some paths to the LD_LIBRARY_PATH
if [[ -z $MTLB_FLDR ]]; then
    if [[ -z $MCR_DIR ]]; then
        MTLB_FLDR=`which matlab 2>/dev/null`
        if [[ $? -ne 0 ]]; then echo "Unable to find MATLAB or MATLAB Compiler Runtime." 1>&2; echo; usage; fi;
        while [ -h "$MTLB_FLDR" ]; do
            DIR="$( cd -P "$( dirname "$MTLB_FLDR" )" && pwd -P )"
            MTLB_FLDR="$(readlink "$MTLB_FLDR")"
            [[ $MTLB_FLDR != /* ]] && MTLB_FLDR="$DIR/$MTLB_FLDR"
        done
        MTLB_FLDR=`dirname "$( cd -P "$( dirname "$MTLB_FLDR" )" && pwd -P )"`
    elif [ ! -d "$MCR_DIR" ]; then echo "MCR_DIR is not a directory." 1>&2; echo; usage;
    else MTLB_FLDR=$MCR_DIR; fi
fi
if [[ ! -d $MTLB_FLDR/bin/glnxa64 ]] || [[ ! -d $MTLB_FLDR/runtime/glnxa64 ]] || [[ ! -d $MTLB_FLDR/sys/os/glnxa64 ]]; then
    echo "Unable to find MATLAB or MATLAB Compiler Runtime (thought we found $MTLB_FLDR but that wasn't it)." 1>&2; echo; usage;
fi
if [ -z $LD_LIBRARY_PATH ]; then
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64;
else
    export LD_LIBRARY_PATH=$MTLB_FLDR/bin/glnxa64:$MTLB_FLDR/runtime/glnxa64:$MTLB_FLDR/sys/os/glnxa64:$LD_LIBRARY_PATH;
fi


# Setup caching
if [ -z $MCR_CACHE_ROOT ]; then
    export MCR_CACHE_ROOT=/tmp/mcr_cache_root_$USER
    mkdir -p $MCR_CACHE_ROOT
fi


# Find where the bash script actually is so we can find the wrapped program
# This is a bit complicated since this script is actually a symlink
# See stackoverflow.com/questions/59895/can-a-bash-script-tell-what-directory-its-stored-in
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
SOURCE="$( cd -P "$( dirname "$SOURCE" )" && pwd -P )"


# Run the main matlab script
$SOURCE/gen_histogram "${INPUT}" "${OUTPUT}";

# Done
exit $?;

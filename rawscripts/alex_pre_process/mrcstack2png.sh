#! /bin/bash

function show_help () {
cat <<-END
mrcstack2png.sh
Usage:
------
    -i | --input (File name)
        MRC stack to be converted to PNG files
    
    -o | --output (Directory name)
        Path to store output PNG files to
    
    -a | --array
        Process in parallel as an array job
    
    -h | --help
        Display this help
END
}

while :; do
    case $1 in
        -h|--help)
            show_help
            exit
            ;;
        -i|--input)
            input=$2
            shift 2
            continue
            ;;
        -o|--output)
            path_out=$2
            shift 2
            continue
            ;;
        -a|--array)
            array=1
            shift 1
            continue
            ;;
        *)
            break
    esac
    shift
done

if [[ ! $input ]] || [[ ! $path_out ]]; then
    printf 'ERROR: options -i and -o must be specified\n\n' >&2
    show_help
    exit 1
fi

source /home/aperez/.bashrc

if [[ ! -d $path_out ]]; then mkdir ${path_out}; fi
mkdir ${path_out}/test ${path_out}/log

#If array is not selected, launch a non-array job using standard mrc2tif. If it is set, launch a parallel job 
#using mrc2tif on single slices only.
if [[ -z ${array+x} ]]; then
    qsub -v file_mrc=${input},path_out=${path_out} -o ${path_out}/log /data/aperez/sge/mrcstack2png.q
else
    Nslices=`${IMOD_DIR}/bin/header -size $input | tr -s ' ' | cut -d ' ' -f4`
    qsub -t 1-${Nslices} -v file_mrc=${input},path_out=${path_out} -o ${path_out}/log /data/aperez/sge/mrcstack2png.q 
fi

#! /bin/bash

function show_help () {
cat <<-END
run_ehs.sh
Usage:
------
    -i | --images (Directory name)
        Path containing input PNG files to be processed   
 
    -r | --reference (Directory name)
        Path containing reference histogram text files
    
    -o | --output (Directory name)
        Path to save output to
    
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
        -i|--images)
            path_images=$2
            shift 2
            continue
            ;;
        -r|--reference)
            path_ref=$2
            shift 2
            continue
            ;;
        -o|--output)
            path_out=$2
            shift 2
            continue
            ;;
        *)
            break
    esac
    shift
done

if [[ ! $path_images ]] || [[ ! $path_ref ]] || [[ ! $path_out ]]; then
    printf 'ERROR: options -i, -r, and -o  must be specified\n\n' >&2
    show_help
    exit 1
fi

if [[ ! -d $path_out ]]; then mkdir ${path_out}; fi
mkdir ${path_out}/log ${path_out}/ehs

Nslices=`ls ${path_images} | wc -l`
qsub -t 1-${Nslices} -v path_images=${path_images},path_ref=${path_ref},path_out=${path_out}/ehs -o ${path_out}/log run_ehs.q 


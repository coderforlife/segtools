#! /bin/bash

function show_help () {
cat <<-END
process_td.sh
Usage:
------
    -i | --input (File name)
        MRC file the training contours were traced on

    -m | --model (File name)
        Model file consisting of two objects. The first object consists of scattered seed points
        marking the center of each training image. The second object consists of closed contours
        representing manual traces of the object of interest.

    -e | --ehs (Directory name)
       Path to output stack of PNGs from EHS      
 
    -o | --output (Directory name)
        Path to save training images and labels to

    -d | --dim (Integer,Integer)
        Dimensions of the training data in X and Y

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
            file_mrc=$2
            shift 2
            continue
            ;;
        -m|--model)
            file_mod=$2
            shift 2
            continue
            ;;
        -e|--ehs)
	    path_ehs=$2
	    shift 2
	    continue
	    ;;
        -o|--output)
	    path_out=$2
	    shift 2
	    continue
	    ;;
	-d|--dim)
	    dim=$2
	    shift 2
	    continue
	    ;;
        *)
            break
    esac
    shift
done

source /home/aperez/.bashrc

if [[ ! $file_mrc ]] || [[ ! $file_mod ]] || [[ ! $path_ehs ]] || [[ ! $path_out ]] || [[ ! $dim ]]; then
    printf 'ERROR: options -i, -m, -e, -o, and -d  must be specified\n\n' >&2
    show_help
    exit 1
fi

if [[ ! -d $path_ehs ]]; then
    printf 'ERROR: the path specified by -e does not exist\n\n' >&2
    show_help
    exit 1
fi

if [[ ! -f $file_mrc ]]; then
    printf 'ERROR: the MRC file specified by -i does not exist\n\n' >&2
    show_help
    exit 1
fi

if [[ ! -f $file_mod ]]; then
    printf 'ERROR: the model file specified by -m does not exist\n\n' >&2
    show_help
    exit 1
fi

Nobj=`${IMOD_DIR}/bin/imodinfo -a $file_mod | grep -m 1 '^imod' | cut -d ' ' -f2`

if [[ $Nobj -ne 2 ]]; then
    printf 'ERROR: the model file specified by -m must contain exactly two objects\n\n' >&2
    show_help
    exit 1
fi

if [[ ! -d $path_out ]]; then mkdir ${path_out}; fi
mkdir ${path_out}/td ${path_out}/tl ${path_out}/log

qsub -v file_mrc=${file_mrc},file_mod=${file_mod},path_ehs=${path_ehs},path_out=${path_out},dim=${dim} -o ${path_out}/log process_td.q

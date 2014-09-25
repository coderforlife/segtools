#! /bin/bash

function show_help () {
cat <<-END
generate_reference.sh
Usage:
------
    -i | --input (Directory name)
        Path containing the stack of PNG files to run EHS on

    -o | --output (Directory name)
        Path to store the reference histogram to

    -f | --fullstack
        Compute the reference histogram as that of the full image stack specified by --input

    -z (Integer)
        Compute the reference histogram as that of a single image in --input whose value is specified here

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
            path_in=$2
            shift 2
            continue
            ;;
        -o|--output)
            path_out=$2
            shift 2
            continue
            ;;
        -f|--fullstack)
            fullstack=1
            shift 1
            continue
            ;;
        -z)
            z=$2
            shift 2
            continue
            ;;
        *)
            break
    esac
    shift
done

if [[ ! $path_in ]] || [[ ! $path_out ]]; then
    printf 'ERROR: options -i and -o must be specified\n\n' >&2
    show_help; exit 1
fi

if [[ ! $fullstack ]] && [[ ! $z ]]; then
    printf 'ERROR: must chose fullstack mode (-f) or single image mode (-z integer)\n\n' >&2
    show_help; exit 1
fi

if [[ ! -d $path_in ]]; then
    printf 'ERROR: directory specified by -i does not exist\n\n' >&2
    show_help; exit 1
fi

if [[ ! -d $path_out ]]; then mkdir ${path_out}; fi
mkdir ${path_out}/log ${path_out}/err ${path_out}/ref

if [[ -z ${fullstack+x} ]]; then
    qsub -v path_in=${path_in},path_out=${path_out}/ref,N1=${z} -o ${path_out}/log generate_reference.q
else
    Nslices=`ls ${path_in} | wc -l`
    qsub -t 1-${Nslices} -v path_in=${path_in},path_out=${path_out}/ref,N1=1 -o ${path_out}/log -e ${path_out}/err generate_reference.q
fi


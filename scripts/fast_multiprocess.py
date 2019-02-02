#!/usr/bin/python3

# --------------------------------------------------
# Script for running parallel processes
# FSL_FAST
#
# This kind of scripts are useful when you want to
# process a large number of images in batch. On binaries
# that do not use all the system cores, it will spread
# the selected batch of images across different instances
# tha will run in parallel.
#
# As a requirement, each image to process has to be inside a
# separate folder and using the same name. For instance:
# DATA/
#     scan1/**/im.nii.gz
#     scan2/**/im.nii.gz
#     ...
#     scanN/**/im.nii.gz
#
# where im.nii.gz is the image to process.
#
# Run fast_multiprocess --help to see available options
# --------------------------------------------------

import os
import glob
import argparse
from multiprocessing import Pool


def extract_scans(input_folder, input_vol):
    """
    Extract a list of scans tagged with the input_vol name.
    """
    pattern = input_folder + '/**/' + input_vol
    selected_files = glob.glob(pattern, recursive=True)

    return sorted(selected_files)

def check_scans(input_files, pattern='_pve_0.nii.gz'):
    """
    Check if a previous output result exists using a pattern as
    control
    """

    output_files = []
    for i in input_files:
        path_, file_ = os.path.split(i)
        file_ = file_.split('.')[0]
        if os.path.exists(os.path.join(path_, file_ + pattern)) is False:
            output_files.append(i)

    return  output_files

def run_batch(options):
    """
    Process the current batch of image scans.
    Options:
    - out_name: set name for output
    - classes: number of segmentation classes
    - modality: choose input modality: t1=1, t2=2, pd=3
    - achive: save all results in a new folder = out_name
    """

    COMMAND = 'fsl5.0-fast -n {} -t {} -o {} {}'

    for b in options['data']:
        print('--> running scan ', b)
        path_, _ = os.path.split(b)
        exp_folder = os.path.join(path_, options['out_name'])
        os.system(COMMAND.format(options['classes'],
                                 options['modality'],
                                 exp_folder,
                                 b))
        if options['archive']:
            if os.path.exists(exp_folder) is False:
                os.mkdir(exp_folder)
            os.system('mv ' + path_ + '/'
                      + options['out_name']
                      + '* ' + exp_folder + '/'
                      + ' 2>/dev/null')

def main(args):
    """
    main script
    """

    options = {}
    options['classes'] = args.classes
    options['modality'] = args.modality
    options['out_name'] = args.output_name
    options['archive'] = args.archive
    input_files = extract_scans(args.data_folder, args.input_name)

    if args.check:
        input_files = check_scans(input_files)

    workers = args.workers
    if workers > len(input_files):
        workers = len(input_files)

    # distribute data across workers
    batches = [[] for w in range(workers)]
    for i, f in enumerate(input_files):
        print('adding ', f , 'to worker', i % workers)
        batches[i % workers].append(f)

    # build working batches with method options + data to process
    working_batches = []
    for b in range(workers):
        c_options = options.copy()
        c_options.setdefault('data', batches[b])
        working_batches.append(c_options)

    # process data in parallel
    p = Pool(workers)
    p.map(run_batch, working_batches)

    return batches

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Using fsl5-fast in parallel")
    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='Number of workers to use in parallel')
    parser.add_argument('--classes',
                        type=int,
                        default=3,
                        help='Number of segmentation classes')
    parser.add_argument('--modality',
                        type=int,
                        default=1,
                        help='Input modality: T1=1, T2=2, PD=3')
    parser.add_argument('--data_folder',
                        type=str,
                        help='Input data folder')
    parser.add_argument('--output_name',
                        type=str,
                        help='Result name. Combined with --archive also sets an output folder where results are stored')
    parser.add_argument('--input_name',
                        type=str,
                        help='Input volume name')
    parser.add_argument('--check',
                        action='store_true',
                        help='check if a previous segmentation exists for each image')
    parser.add_argument('--archive',
                        action='store_true',
                        help='archive results in a ouput folder = output_name')

    args = parser.parse_args()
    batches_ = main(args)

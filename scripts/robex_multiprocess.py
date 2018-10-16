#!/usr/bin/python3.6

# --------------------------------------------------
# Script for running parallel processes
# ROBEX
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


def process_batch(data_batch, process='fsl'):
    """
    Process the current batch of image scans.
    Output images are saved with the same input name.
    """
    COMMAND = '~/bin/ROBEX/runROBEX.sh {} {}'
    for b in data_batch:
        print('--> running scan', b)
        os.system(COMMAND.format(b, b.replace('.nii.gz', '_brain.nii.gz')))

def main(args):
    """
    main script
    """

    input_folder = args.data_folder
    input_scan = args.input_name
    workers = args.workers
    input_files = extract_scans(input_folder, input_scan)
    batches = [input_files[c * workers:c * workers + workers]
               for c, r in enumerate(range(0, len(input_files), len(input_files) // workers))]

    # process data in parallel
    p = Pool(workers)
    p.map(process_batch, batches)


if __name__ == '__main__':

    # load options from input
    parser = argparse.ArgumentParser(description="Using ROBEx in parallel")
    parser.add_argument('--workers',
                        type=int,
                        default=10,
                        help='Number of workers to use in parallel')
    parser.add_argument('--data_folder',
                        type=str,
                        help='Data folder')
    parser.add_argument('--output_name',
                        type=str,
                        help='Output volume name')
    parser.add_argument('--input_name',
                        type=str,
                        help='Input volume name')

    args = parser.parse_args()

    main(args)

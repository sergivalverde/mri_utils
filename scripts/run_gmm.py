import numpy as np
import os
import nibabel as nib
from sklearn.mixture import
import sys
sys.path.append('../')
from mri_utils.processing import gmm_clustering

# --------------------------------------------------
# configure options
# --------------------------------------------------

NCLASSES = 5
IMPATH = '/home/sergivalverde/DATA/VH2020/processing/MNI/2016'
MODS = ['flair', 't2', 't1', 'dp']
TIMEPOINTS = ['basal', 'followup']


images = sorted(os.listdir(IMPATH))
for im in images:
    print('Processing scan', im)

    for m in MODS:
        for t in TIMEPOINTS:

            current_scan = m + '_' + t + '.nii.gz'
            print('         processing', current_scan)
            scan = nib.load(
                os.path.join(IMPATH, im, 'processed', current_scan))
            scan_img = scan.get_fdata()

            probs, labels = gmm_clustering(scan_img, NCLASSES, 500)

            out_path = os.path.join(IMPATH,
                                    im,
                                    'synthesis',
                                    m + '_' + t + '_gmm_5c')

            if not os.path.exists(out_path):
                os.mkdir(out_path)

            for i, r in enumerate(probs):
                r_ = nib.Nifti1Image(r, affine=scan.affine)
                r_.to_filename(
                    os.path.join(out_path, 'c' + str(i) + '.nii.gz'))

            r_ = nib.Nifti1Image(labels.astype('<f4'), affine=scan.affine)
            r_.to_filename(os.path.join(out_path, 'discrete_seg_5c.nii.gz'))

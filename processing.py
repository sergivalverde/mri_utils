# -------------------------------------------------------
# Image processing functions
# Useful for brain MRI analysis
#
# Sergi Valverde 2018
# svalverde@eia.udg.edu
#
# -------------------------------------------------------

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
import SimpleITK as sitk


def filter_regions_volume(input_mask, threshold=0.5, min_volume=10):
    """
    Remove regions with region volume < min_volume. Optionally, an initial
    threshold is applied to binarize probabilities before filtering.

    Inputs:
    - mask: 3D np.ndarray
    - threshold: binarize parameter (def: 0.5)
    - min_volume: Minimum region size used for filtering (in voxels) (def: 10)

    Output:
    - mask: 3D np.ndarray mask > threshold where regions < 10 have been
      filtered
    """

    mask = input_mask >= threshold
    regions, num_regions = label(mask)
    labels = np.arange(1, num_regions+1)
    output_mask = np.zeros_like(mask)
    prob_mask = np.zeros(mask.shape)

    if num_regions > 0:
        region_vol = lc(mask, regions, labels, np.sum, int, 0)
        for l in labels:
            if region_vol[l-1] > min_volume:
                current_voxels = np.stack(np.where(regions == l), axis=1)
                int_mean = np.median(input_mask[current_voxels[:, 0],
                                              current_voxels[:, 1],
                                              current_voxels[:, 2]])
                output_mask[current_voxels[:, 0],
                            current_voxels[:, 1],
                            current_voxels[:, 2]] = 1
                prob_mask[current_voxels[:, 0],
                          current_voxels[:, 1],
                          current_voxels[:, 2]] = int_mean

    return output_mask, prob_mask


def histogram_matching(mov_scan, ref_scan,
                       histogram_levels=2048,
                       match_points=100,
                       set_th_mean=True):
    """
    Histogram matching following the method developed on
    Nyul et al 2001 (ITK implementation)

    inputs:
    - mov_scan: np.array containing the image to normalize
    - ref_scan np.array containing the reference image
    - histogram levels
    - number of matched points
    - Threshold Mean setting

    outputs:
    - histogram matched image
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    # perform histogram matching
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref.GetPixelID())

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(set_th_mean)
    matched_vol = matcher.Execute(mov, ref)

    return sitk.GetArrayFromImage(matched_vol)


def n4_normalization(input_scan, max_iters=400, levels=1):
    """
    N4ITK: Improved N3 Bias Correction, Tustison et al. 2010)

    inputs:
    - input scan: np.array containing the image to process
    - max_iters: number of processing iterations
    - levels

    outputs:
    - bias corrected image
    """

    # convert np array into itk image objects
    scan = sitk.GetImageFromArray(input_scan.astype('float32'))

    # process the input image
    mask = sitk.OtsuThreshold(scan, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([max_iters] * levels)
    output = corrector.Execute(scan, mask)

    return sitk.GetArrayFromImage(output)


def normalize_data(im,
                   norm_type='standard',
                   brainmask=None,
                   datatype=np.float32):
    """
    Zero mean normalization

    inputs:
    - im: input data
    - nomr_type: 'zero_one', 'standard'

    outputs:
    - normalized image
    """
    mask = np.copy(im > 0 if brainmask is None else brainmask)

    if norm_type == 'standard':
        im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
        im = im / im[np.nonzero(im)].std()

    if norm_type == 'zero_one':
        min_int = abs(im.min())
        max_int = im.max()
        if im.min() < 0:
            im = im.astype(dtype=datatype) + min_int
            im = im / (max_int + min_int)
        else:
            im = (im.astype(dtype=datatype) - min_int) / max_int

    # do not apply normalization to non-brain parts
    # im[mask==0] = 0
    return im

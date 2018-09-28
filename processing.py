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

def filter_regions_volume(mask, threshold=0.5, min_volume=10):
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

    mask = mask >= threshold
    regions, num_regions = label(mask)
    labels = np.arange(1, num_regions+1)
    output_mask = np.zeros_like(mask)

    if num_regions > 0:
        region_vol = lc(mask, regions, labels, np.sum, int, 0)
        for l in labels:
            if region_vol[l-1] > min_volume:
                current_voxels = np.stack(np.where(regions == l), axis=1)
                output_mask[current_voxels[:, 0],
                            current_voxels[:, 1],
                            current_voxels[:, 2]] = 1

    return output_mask


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
    ref = sitk.GetImageFromArray(ref_scan)
    mov = sitk.GetImageFromArray(mov_scan)

    # perform histogram matching
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(ref.GetPixelID())

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(histogram_levels)
    matcher.SetNumberOfMatchPoints(match_points)
    matcher.SetThresholdAtMeanIntensity(set_th_mean)
    matched_vol = matcher.Execute(mov, ref)

    return sitk.GetArrayFromImage(matched_vol)

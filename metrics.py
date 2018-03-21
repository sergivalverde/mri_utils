# -------------------------------------------------------
# Evaluation metrics for MRI:
# implements most of evaluation metrics used in brain MRI
#
# Sergi Valverde 2018
# svalverde@eia.udg.edu
#
# -------------------------------------------------------

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
from scipy.ndimage.morphology import binary_erosion as imerode
from sklearn.neighbors import NearestNeighbors


def regionprops(mask):
    """
    Get region properties
    """
    regions, num_regions = label(as_logical(mask))
    labels = np.arange(1, num_regions+1)
    areas = lc(regions > 0, regions, labels, np.sum, int, 0)
    return regions, labels, areas


def as_logical(mask):
    """
    convert to Boolean
    """
    return np.array(mask).astype(dtype=np.bool)


def num_regions(mask):
    """
    compute the number of regions from an input mask
    """
    regions, num_regions = label(as_logical(mask))
    return num_regions


def num_voxels(mask):
    """
    compute the number of voxels from an input mask
    """
    return np.sum(as_logical(mask))


def true_positive_seg(gt, mask):
    """
    compute the number of true positive voxels between a input mask an
    a ground truth (GT) mask
    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(a, b))


def true_positive_det(gt, mask):
    """
    compute the number of positive regions between a input mask an
    a ground truth (GT) mask
    """
    regions, num_regions = label(as_logical(gt))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    tpr = lc(mask, regions, labels, np.sum, int, 0)

    return np.sum(tpr > 0)


def false_negative_seg(gt, mask):
    """
    compute the number of false negative voxels
    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(a, np.logical_not(b)))


def false_negative_det(gt, mask):
    """
    compute the number of false negative regions between a input mask an
    a ground truth (GT) mask
    """
    regions, num_regions = label(as_logical(gt))
    labels = np.arange(1, num_regions+1)
    mask = as_logical(mask)
    tpr = lc(mask, regions, labels, np.sum, int, 0)

    return np.sum(tpr == 0)


def false_positive_seg(gt, mask):
    """
    compute the number of false positive voxels between a input mask an
    a ground truth (GT) mask
    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(np.logical_not(a), b))


def false_positive_det(gt, mask):
    """
    compute the number of false positive regions between a input mask an
    a ground truth (GT) mask
    """
    regions, num_regions = label(as_logical(mask))

    labels = np.arange(1, num_regions+1)
    gt = as_logical(gt)

    return np.sum(lc(gt, regions, labels, np.sum, int, 0) == 0) \
        if num_regions > 0 else 0


def true_negative_seg(gt, mask):
    """
    compute the number of true negative samples between an input mask and
    a ground truth mask
    """
    a = as_logical(gt)
    b = as_logical(mask)

    return np.count_nonzero(np.logical_and(np.logical_not(a),
                                           np.logical_not(b)))


def TPF_seg(gt, mask):
    """
    compute the True Positive Fraction (Sensitivity) between an input mask and
    a ground truth mask
    """
    TP = true_positive_seg(gt, mask)
    GT_voxels = np.sum(as_logical(gt)) if np.sum(as_logical(gt)) > 0 else 0

    return float(TP) / GT_voxels


def TPF_det(gt, mask):
    """
    Compute the TPF (sensitivity) detecting candidate regions between an
    input mask and a ground truth mask
    """

    TP = true_positive_det(gt, mask)
    number_of_regions = num_regions(gt)

    return float(TP) / number_of_regions


def FPF_seg(gt, mask):
    """
    Compute the False positive fraction between an input mask and a ground
    truth mask
    """
    b = num_voxels(mask)
    fpf = 100.0 * false_positive_seg(gt, mask) / b if b > 0 else 0

    return fpf


def FPF_det(gt, mask):
    """
    Compute the FPF detecting candidate regions between an
    input mask and a ground truth mask
    """

    FP = false_positive_det(gt, mask)
    number_of_regions = num_regions(mask)

    return float(FP) / number_of_regions if number_of_regions > 0 else 0


def DSC_seg(gt, mask):
    """
    Compute the Dice (DSC) coefficient betweeen an input mask and a ground
    truth mask
    """
    A = num_voxels(gt)
    B = num_voxels(mask)

    return 2.0 * true_positive_seg(gt, mask) / (A + B) \
        if (A + B) > 0 else 0


def AOV_seg(gt, mask):
    """
    Compute the Area Overlap coefficient betweeen an input mask and a ground
    truth mask == TPF
    """
    a = np.sum(np.logical_and(as_logical(gt), as_logical(mask)))
    b = num_voxels(as_logical(gt))

    return float(a) / b


def DSC_det(gt, mask):
    """
    Compute the Dice (DSC) coefficient betweeen an input mask and a ground
    truth mask
    """
    A = num_regions(gt)
    B = num_regions(mask)

    return 2.0 * true_positive_det(gt, mask) / (A + B) \
        if (A + B) > 0 else 0


def PVE(gt, mask, type='absolute'):
    """
    Compute the volume difference error betweeen an input mask and a ground
    truth mask

    type parameter controls if the error is relative or absolute
    """
    A = num_voxels(gt)
    B = num_voxels(mask)

    if type == 'absolute':
        pve = np.abs(float(B - A) / A)
    else:
        pve = float(B - A) / A

    return pve


def PPV_det(gt, mask):
    """
    Compute the positive predictive value (recall) for the detected
    regions between an input mask and a ground truth mask
    """

    a = TPF_det(gt, mask)
    b = TPF_det(gt, mask) + FPF_det(gt, mask)

    return a / b if a > 0 else 0


def PPV_seg(gt, mask):
    """
    Compute the positive predictive value (recall) for the detected
    regions between an input mask and a ground truth mask
    """
    a = TPF_seg(gt, mask)
    b = TPF_seg(gt, mask) + FPF_seg(gt, mask)

    return a / b if a > 0 else 0


def f_score(gt, mask):
    """
    Compute a custom score between an input mask and a ground truth mask

    F = 3 * DSC_s + TPF_d + (1- FPF) / DSC_s +TPF_d + (1-FPF)
    """

    a = 3.0 * DSC_seg(gt, mask) * TPF_det(gt, mask) * (1 - FPF_seg(gt, mask))
    b = DSC_seg(gt, mask) + TPF_det(gt, mask) + (1 - FPF_seg(gt, mask))

    return a / b if a > 0 else 0


def eucl_distance(a, b):
    """
    Euclidian distance between a and b
    """
    nbrs_a = NearestNeighbors(n_neighbors=1,
                              algorithm='kd_tree').fit(a) if a.size > 0 else None
    nbrs_b = NearestNeighbors(n_neighbors=1,
                              algorithm='kd_tree').fit(b) if b.size > 0 else None
    distances_a, _ = nbrs_a.kneighbors(b) if nbrs_a and b.size > 0 else ([np.inf], None)
    distances_b, _ = nbrs_b.kneighbors(a) if nbrs_b and a.size > 0 else ([np.inf], None)

    return [distances_a, distances_b]


def surface_distance(gt, mask, spacing=list((1, 1, 3))):
    """
    Compute the surface distance between the input mask and a
    ground truth mask

    - spacing: sets the input resolution
    """
    a = as_logical(gt)
    b = as_logical(mask)
    a_bound = np.stack(np.where(
        np.logical_and(a, np.logical_not(imerode(a)))), axis=1) * spacing
    b_bound = np.stack(np.where(
        np.logical_and(b, np.logical_not(imerode(b)))), axis=1) * spacing
    return eucl_distance(a_bound, b_bound)


def mask_distance(gt, mask, spacing=list((1, 1, 3))):
    """
    Compute the mask distance between the input mask and the
    ground truth mask

    - spacing: sets the input resolution
    """
    a = as_logical(gt)
    b = as_logical(mask)
    a_full = np.stack(np.where(a), axis=1) * spacing
    b_full = np.stack(np.where(b), axis=1) * spacing
    return eucl_distance(a_full, b_full)


def ASD(gt, mask, spacing):
    """
    Compute the average_surface_distance between an input mask and a
    ground truth mask

    - spacing: sets the input resolution
    """
    distances = np.concatenate(surface_distance(gt, mask, spacing))
    return np.mean(distances)


def HD(gt, mask, spacing):
    """
    Compute the Haursdoff distance between an input mask and a
    groud truth mask

    - spacing: sets the input resolution
    """
    distances = surface_distance(gt, mask, spacing)
    return np.max([np.max(distances[0]), np.max(distances[1])])


def MHD(gt, mask, spacing):
    """
    Compute the modified Haursdoff distance between an input mask and a
    groud truth mask using the spacing parameter

    - spacing: sets the input resolution
    """
    distances = mask_distance(gt, mask, spacing)
    return np.max([np.mean(distances[0]), np.mean(distances[1])])

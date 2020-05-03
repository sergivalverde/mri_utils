# -------------------------------------------------------
# Image processing functions
# Useful for brain MRI analysis
#
# Sergi Valverde 2018
# svalverde@eia.udg.edu
#
# -------------------------------------------------------

import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.ndimage import labeled_comprehension as lc
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
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


def nyul_apply_standard_scale(input_image,
                              standard_hist,
                              input_mask=None,
                              interp_type='linear'):
    """

    Based on J.Reinhold code:
    https://github.com/jcreinhold/intensity-normalization

    Use Nyul and Udupa method ([1,2]) to normalize the intensities
    of a MRI image passed as input.

    Args:
        input_image (np.ndarray): input image to normalize
        standard_hist (str): path to output or use standard histogram landmarks
        input_mask (nii): optional brain mask

    Returns:
        normalized (np.ndarray): normalized input image

    References:
        [1] N. Laszlo G and J. K. Udupa, “On Standardizing the MR Image
            Intensity Scale,” Magn. Reson. Med., vol. 42, pp. 1072–1081,
            1999.
        [2] M. Shah, Y. Xiao, N. Subbanna, S. Francis, D. L. Arnold,
            D. L. Collins, and T. Arbel, “Evaluating intensity
            normalization on MRIs of human brain with multiple sclerosis,”
            Med. Image Anal., vol. 15, no. 2, pp. 267–282, 2011.
    """

    # load learned standard scale and the percentiles
    standard_scale, percs = np.load(standard_hist)

    # apply transformation to image
    return do_hist_normalization(input_image,
                                 percs,
                                 standard_scale,
                                 input_mask,
                                 interp_type=interp_type)


def get_landmarks(img, percs):
    """
    get the landmarks for the Nyul and Udupa norm method for a specific image

    Based on J.Reinhold code:
    https://github.com/jcreinhold/intensity-normalization

    Args:
        img (nibabel.nifti1.Nifti1Image): image on which to find landmarks
        percs (np.ndarray): corresponding landmark percentiles to extract

    Returns:
        landmarks (np.ndarray): intensity values corresponding to percs in img
    """
    landmarks = np.percentile(img, percs)
    return landmarks


def nyul_train_standard_scale(img_fns,
                              mask_fns=None,
                              i_min=1,
                              i_max=99,
                              i_s_min=1,
                              i_s_max=100,
                              l_percentile=10,
                              u_percentile=90,
                              step=10):
    """
    determine the standard scale for the set of images

    Based on J.Reinhold code:
    https://github.com/jcreinhold/intensity-normalization


    Args:
        img_fns (list): set of NifTI MR image paths which are to be normalized
        mask_fns (list): set of corresponding masks (if not provided, estimated)
        i_min (float): minimum percentile to consider in the images
        i_max (float): maximum percentile to consider in the images
        i_s_min (float): minimum percentile on the standard scale
        i_s_max (float): maximum percentile on the standard scale
        l_percentile (int): middle percentile lower bound (e.g., for deciles 10)
        u_percentile (int): middle percentile upper bound (e.g., for deciles 90)
        step (int): step for middle percentiles (e.g., for deciles 10)

    Returns:
        standard_scale (np.ndarray): average landmark intensity for images
        percs (np.ndarray): array of all percentiles used
    """

    # compute masks is those are not entered as a parameters
    mask_fns = [None] * len(img_fns) if mask_fns is None else mask_fns

    percs = np.concatenate(([i_min],
                            np.arange(l_percentile, u_percentile+1, step),
                            [i_max]))
    standard_scale = np.zeros(len(percs))

    # process each image in order to build the standard scale
    for i, (img_fn, mask_fn) in enumerate(zip(img_fns, mask_fns)):
        print('processing scan ', img_fn)
        img_data = nib.load(img_fn).get_data()
        mask = nib.load(mask_fn) if mask_fn is not None else None
        mask_data = img_data > img_data.mean() \
            if mask is None else mask.get_data()
        masked = img_data[mask_data > 0]
        landmarks = get_landmarks(masked, percs)
        min_p = np.percentile(masked, i_min)
        max_p = np.percentile(masked, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])
        landmarks = np.array(f(landmarks))
        standard_scale += landmarks
    standard_scale = standard_scale / len(img_fns)
    return standard_scale, percs


def do_hist_normalization(input_image,
                          landmark_percs,
                          standard_scale,
                          mask=None,
                          interp_type='linear'):
    """
    do the Nyul and Udupa histogram normalization routine with a given set of
    learned landmarks

    Based on J.Reinhold code:
    https://github.com/jcreinhold/intensity-normalization


    Args:
        img (np.ndarray): image on which to find landmarks
        landmark_percs (np.ndarray): corresponding landmark points of standard scale
        standard_scale (np.ndarray): landmarks on the standard scale
        mask (np.ndarray): foreground mask for img

    Returns:
        normalized (np.ndarray): normalized image
    """

    mask_data = input_image > input_image.mean() if mask is None else mask
    masked = input_image[mask_data > 0]
    landmarks = get_landmarks(masked, landmark_percs)
    if interp_type == 'linear':
        f = interp1d(landmarks, standard_scale, fill_value='extrapolate')

    # apply transformation to input image
    return f(input_image)


def gmm_clustering(img_data, n_classes=3, max_iter=50, brainmask=None):
    """
    GMM clustering

    inputs:
    - img_data: np.array containing the image to process
    - n_classes: number of classes
    - max_iter: number of iterations to perform
    - brainmask: input brainmask

    outputs:
    - res_array: np.array n_classes x img_data with output probabilities
    - labels: np.array with resulting class labels
    """
    if brainmask is None:
        brainmask = img_data > 0

    brain = img_data
    brain = np.expand_dims(img_data[brainmask].flatten(), 1)
    gmm = GaussianMixture(n_classes, max_iter=max_iter)
    gmm.fit(brain)
    classes_ = np.argsort(gmm.means_.T.squeeze())

    # predict probabilities
    res = gmm.predict_proba(brain)
    res_array = []
    for c in classes_:
        prob_class = np.zeros_like(img_data)
        prob_class[brainmask > 0] = res[:, c]
        res_array.append(prob_class)

    # predict labels
    labels = np.array(res_array)
    labels = np.argmax(res_array, axis=0)
    labels[brainmask > 0] += 1

    return res_array, labels

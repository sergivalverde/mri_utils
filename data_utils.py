# --------------------------------------------------
#
# data utils to extract patches from MRI images
#
# Sergi Valverde 2018
#
# --------------------------------------------------

import numpy as np
from operator import add


def extract_patches(input_image,
                    roi=None,
                    voxel_coords=None,
                    patch_size=(15, 15, 15),
                    step_size=(1, 1, 1)):
    """
    Extract patches of size patch_size from an input image given as input

    inputs:
    - input_image:  3D np.array
    - roi: region of interest to extract samples. input_image > 0 if not set
    - voxel_coords: Already computed voxel coordenades
    - patch_size: output patch size
    - step_size: sampling overlap in x, y and z

    output:
    - list of sampled patches: 4D array [n, patch_size] eg: (100, 15, 15 ,15)
    - list of voxel_coordenates
    """

    # check roi
    if roi is None:
        roi = input_image > 0

    # get voxel coordenates taking into account step sampling if those are
    # not passed as input
    if voxel_coords is None:
        voxel_coords = get_voxel_coordenates(input_image,
                                             roi,
                                             step_size=step_size)

    # extract patches based on the sampled voxel coordenates
    out_patches = get_patches(input_image, voxel_coords, patch_size)

    return out_patches, voxel_coords


def get_voxel_coordenates(input_data,
                          roi,
                          random_pad=(0, 0, 0),
                          step_size=(1, 1, 1)):
    """
    Get voxel coordenates based on a sampling step size or input mask.
    For each selected voxel, return its (x,y,z) coordinate.

    inputs:
    - input_data (useful for extracting non-zero voxels)
    - roi: region of interest to extract samples. input_data > 0 if not set
    - step_size: sampling overlap in x, y and z
    - random_pad: initial random padding applied to indexes

    output:
    - list of voxel coordenates
    """

    # compute initial padding
    r_pad = np.random.randint(random_pad[0]+1) if random_pad[0] > 0 else 0
    c_pad = np.random.randint(random_pad[1]+1) if random_pad[1] > 0 else 0
    s_pad = np.random.randint(random_pad[2]+1) if random_pad[2] > 0 else 0

    # precompute the sampling points based on the input
    sampled_data = np.zeros_like(input_data)
    for r in range(r_pad, input_data.shape[0], step_size[0]):
        for c in range(c_pad, input_data.shape[1], step_size[1]):
            for s in range(s_pad, input_data.shape[2], step_size[2]):
                sampled_data[r, c, s] = 1

    # apply sampled points to roi and extract sample coordenates
    # [x, y, z] = np.where(input_data * roi * sampled_data)
    [x, y, z] = np.where(roi * sampled_data)

    # return as a list of tuples
    return [(x_, y_, z_) for x_, y_, z_ in zip(x, y, z)]


def get_patches(input_data, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of voxel coordenates

    inputs:
    - input_data: a tridimensional np.array matrix
    - centers:  centre voxel coordenate for each patch
    - patch_size: patch size (x,y,z)

    outputs:
    - patches: np.array containing each of the patches
    """
    # If the size has even numbers, the patch will be centered. If not,
    # it will try to create an square almost centered. By doing this we allow
    # pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]

    if list_of_tuples and sizes_match:
        # apply padding to the input image and re-compute the voxel coordenates
        # according to the new dimension
        padded_image = apply_padding(input_data, patch_size)
        patch_half = tuple([idx // 2 for idx in patch_size])
        new_centers = [map(add, center, patch_half) for center in centers]
        # compute patch locations
        slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]

        # extact patches
        patches = [padded_image[idx] for idx in slices]

    return np.array(patches)


def reconstruct_image(input_data, centers, output_size):
    """
    Reconstruct image based on several ovelapping patch samples

    inputs:
    - input_data: a np.array list with patches
    - centers: center voxel coordenates for each patch
    - output_size: output image size (x,y,z)

    outputs:
    - reconstructed image
    """

    # apply a padding around edges before writing the results
    # recompute the voxel dimensions
    patch_size = input_data[0, :].shape
    out_image = apply_padding(np.zeros(output_size), patch_size)
    patch_half = tuple([idx // 2 for idx in patch_size])
    new_centers = [map(add, center, patch_half) for center in centers]
    # compute patch locations
    slices = [[slice(c_idx-p_idx, c_idx+s_idx-p_idx)
               for (c_idx, p_idx, s_idx) in zip(center,
                                                patch_half,
                                                patch_size)]
              for center in new_centers]

    # for each patch, sum it to the output patch and
    # then update the frequency matrix

    freq_count = np.zeros_like(out_image)

    for patch, slide in zip(input_data, slices):
        out_image[slide] += patch
        freq_count[slide] += np.ones(patch_size)

    # invert the padding applied for patch writing
    out_image = invert_padding(out_image, patch_size)
    freq_count = invert_padding(freq_count, patch_size)

    # the reconstructed image is the mean of all the patches
    out_image /= freq_count
    out_image[np.isnan(out_image)] = 0

    return out_image


def apply_padding(input_data, patch_size, mode='constant', value=0):
    """
    Apply padding to edges in order to avoid overflow

    """

    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    padded_image = np.pad(input_data,
                          padding,
                          mode=mode,
                          constant_values=value)

    return padded_image


def invert_padding(padded_image, patch_size):
    """
    Invert paadding on edges to recover the original shape

    inputs:
    - padded_image defined by apply_padding function
    - patch_size (x,y,z)

    """

    patch_half = tuple([idx // 2 for idx in patch_size])
    padding = tuple((idx, size-idx)
                    for idx, size in zip(patch_half, patch_size))

    return padded_image[padding[0][0]:-padding[0][1],
                        padding[1][0]:-padding[1][1],
                        padding[2][0]:-padding[2][1]]

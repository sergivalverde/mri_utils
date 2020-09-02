import numpy as np


def uniform_sampling(roi_mask, num_samples):
    """
    Uniform sampling strategy: extract uniformily a particular
    number of samples from the image.

    Inputs:

    - roi mask: input ROI mask to extract samples from
    - num samples: Number of samples per image to use

    Outputs:
    - x, y, z: list containing voxel indices
    """

    num_roi_voxels = np.sum(roi_mask == 1)
    x, y, z = np.where(roi_mask > 0)

    if num_samples < num_roi_voxels:
        int_sampling = int(num_roi_voxels / num_samples)
        sampling_list = np.arange(0,
                                  num_roi_voxels,
                                  int_sampling)
        x = x[sampling_list]
        y = y[sampling_list]
        z = z[sampling_list]

    else:
        expand_interval = int(num_samples / num_roi_voxels) + 1
        x = np.repeat(x, expand_interval)
        y = np.repeat(y, expand_interval)
        z = np.repeat(z, expand_interval)
        x = x[:num_samples]
        x = y[:num_samples]
        x = z[:num_samples]

    return x, y, z


def binary_balanced_sampling(label_mask,
                             roi_mask,
                             patch_size=(32, 32, 32),
                             apply_offset=False):

    """
    balanced sampling strategy: extract the same number of samples
    from positive and negative classes

    Inputs:

    - label mask: Label image with positive samples
    - roi mask: input ROI mask to extract samples from
    - patch size: patch size, by default: 32, 32, 32
    - apply offset to samples

    Outputs:
    - x, y, z: list containing voxel indices
    """

    num_pos_voxels = np.sum(label_mask > 0)
    roi_mask[label_mask == 1] = 0

    brain_voxels = np.stack(np.where(roi_mask > 0), axis=1)
    sampled_mask = np.copy(label_mask)
    for voxel in np.random.permutation(brain_voxels)[:num_pos_voxels]:
        sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

    x, y, z = np.where(sampled_mask == 1)

    if apply_offset:
        x, y, z = __apply_offset(x, y, z, sampled_mask.shape, patch_size)

    return x, y, z


def binary_hybrid_sampling(label_mask,
                           roi_mask,
                           num_samples=2000,
                           patch_size=(32, 32, 32),
                           apply_offset=False):
    """
    Hybrid sampling strategy: extract a number positive samples and the
    same number of negative samples uniformily.

    Inputs:

    - label mask: Label image with positive samples
    - roi mask: input ROI mask to extract samples from
    - patch size: patch size, by default: 32, 32, 32
    - num samples: Number of samples per image to use
    - apply offset to samples

    Outputs:
    - x, y, z: list containing voxel indices
    """

    # positive samples
    # sample voxels randomly until size equals self.num_samples
    x, y, z = np.where(label_mask > 0)
    pos_samples = len(x)

    if pos_samples < num_samples:
        expand_interval = int(num_samples / pos_samples) + 1
        x = np.repeat(x, expand_interval)
        y = np.repeat(y, expand_interval)
        z = np.repeat(z, expand_interval)

    else:
        expand_interval = int(pos_samples / num_samples)
        indexes = np.arange(0, pos_samples, expand_interval)
        x = x[indexes]
        y = y[indexes]
        z = z[indexes]

    x = x[:num_samples]
    y = y[:num_samples]
    z = z[:num_samples]

    if apply_offset:
        x, y, z = __apply_offset(x, y, z, roi_mask.shape, patch_size)

    x_p = np.copy(x)
    y_p = np.copy(y)
    z_p = np.copy(z)

    # negative samples

    # roi_mask[label_mask == 1] = 0
    negative_voxels = np.sum(roi_mask == 1)
    x, y, z = np.where(roi_mask > 0)

    if num_samples < negative_voxels:
        int_sampling = int(negative_voxels / num_samples)
        sampling_list = np.arange(0, negative_voxels, int_sampling)
        x = x[sampling_list]
        y = y[sampling_list]
        z = z[sampling_list]

    else:
        expand_interval = int(num_samples / negative_voxels) + 1
        x = np.repeat(x, expand_interval)
        y = np.repeat(y, expand_interval)
        z = np.repeat(z, expand_interval)

    x = x[:num_samples]
    x = y[:num_samples]
    x = z[:num_samples]

    x = np.concatenate([x_p, x])
    y = np.concatenate([y_p, y])
    z = np.concatenate([z_p, z])

    return x, y, z


def __apply_offset(x, y, z, roi_shape, patch_size=(32, 32, 32)):
    """
    Apply offset to sampled voxels

    Input:
    - x, y, z voxel coordenates
    - roi shape
    - patch size = (32, 32, 32)


    """

    patch_half = tuple([idx // 2 for idx in patch_size])

    min_int_x = - patch_half[0] + 1
    max_int_x = patch_half[0] - 1
    min_int_y = - patch_half[1] + 1
    max_int_y = patch_half[1] - 1
    min_int_z = - patch_half[2] + 1
    max_int_z = patch_half[2] - 1
    x += np.random.randint(low=min_int_x,
                           high=max_int_x,
                           size=x.shape)
    y += np.random.randint(low=min_int_y,
                           high=max_int_y,
                           size=y.shape)
    z += np.random.randint(low=min_int_z,
                           high=max_int_z,
                           size=z.shape)

    # check boundaries
    x = np.maximum(patch_half[0], x)
    x = np.minimum(roi_shape[0] - patch_half[0], x)
    y = np.maximum(patch_half[1], y)
    y = np.minimum(roi_shape[1] - patch_half[1], y)
    z = np.maximum(patch_half[2], z)
    z = np.minimum(roi_shape[2] - patch_half[2], z)

    return x, y, z

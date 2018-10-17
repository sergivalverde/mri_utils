
import SimpleITK as sitk

# using ITK procedures

def rigid_transformation(ref_scan,
                         mov_scan,
                         number_bins=50,
                         levels=3,
                         steps=50,
                         sampling=0.5,
                         learning_rate=1.0,
                         min_step=0.0001,
                         max_step=0.2,
                         relaxation_factor=0.5,
                         verbose=1):
    """
    Compute a rigid trasnformation between a ref and a moving scan.

    inputs:
       - ref_scan: numpy 3D image containing the reference image
       - mov_scan: numpy 3D image containing the moving image
       - number_bins: number of histogram bins (50)
       - levels: number of levels used (3)
       - steps: Steps per level (50)
       - sampling: (0.5)
       - learning_rate: (1.0)
       - min_step: (0.001)
       - max_step: (0.2)
       - relaxation_factor: (0.5)
       - verbose: print stuff

    outputs:
       - transf: itk trasnformation
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    # compute transformations
    transf = sitk.CenteredTransformInitializer(
        ref,
        mov,
        sitk.VersorRigid3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS)

    # Registration parameters
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Connect all of the observers so that we can perform plotting during registration.
    if verbose > 0:
        print('Rigid initial registration')
        registration.AddCommand(sitk.sitkMultiResolutionIterationEvent,
                                lambda: print('{:} level {:}'.format(
                                    registration.GetName(),
                                    registration.GetCurrentLevel())))

    if verbose > 1:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print_current(registration, transf))

    # Initial versor optimisation
    registration.SetInitialTransform(transf)
    registration.Execute(ref, mov)

    return transf


def affine_transformation(
        ref_scan,
        mov_scan,
        initial_tf,
        number_bins=50,
        levels=3,
        steps=50,
        sampling=0.5,
        learning_rate=1.0,
        min_step=0.0001,
        max_step=0.2,
        relaxation_factor=0.5,
        verbose=1):
    """
    Compute a affine transformation between a ref and a moving scan.

    inputs:
       - ref_scan: numpy 3D image containing the reference image
       - mov_scan: numpy 3D image containing the moving image
       - initial_tf: initial rigid transformation
       - number_bins: number of histogram bins (50)
       - levels: number of levels used (3)
       - steps: Steps per level (50)
       - sampling: (0.5)
       - learning_rate: (1.0)
       - min_step: (0.001)
       - max_step: (0.2)
       - relaxation_factor: (0.5)
       - verbose: print stuff

    outputs:
       - transf: itk trasnformation
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    optimized_tf = sitk.AffineTransform(3)

    # Registration parameters
    registration = sitk.ImageRegistrationMethod()

    # Similarity metric settings
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_bins)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling)
    registration.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=steps,
        maximumStepSizeInPhysicalUnits=max_step,
        relaxationFactor=relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    smoothing_sigmas = range(levels - 1, -1, -1)
    shrink_factor = [2**i for i in smoothing_sigmas]
    registration.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factor)
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # affine Optimizer settings.
    registration.RemoveAllCommands()
    if verbose > 0:
        print('\tAffine registration')
        registration.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: print('\t > %s level %d' % (
                registration.GetName(),
                registration.GetCurrentLevel()
            ))
        )
    if verbose > 1:
        registration.AddCommand(
            sitk.sitkIterationEnvent,
            lambda: print_current(registration, optimized_tf)
        )

    registration.SetMovingInitialTransform(initial_tf)
    registration.SetInitialTransform(optimized_tf)

    registration.Execute(ref, mov)

    affine_tf = sitk.Transform(optimized_tf)
    affine_tf.AddTransform(initial_tf)

    return affine_tf


def linear_registration(ref_scan,
                        mov_scan,
                        reg_type='affine',
                        interpolation=sitk.sitkBSpline,
                        number_bins=50,
                        levels=3,
                        steps=50,
                        sampling=0.5,
                        learning_rate=1.0,
                        min_step=0.0001,
                        max_step=0.2,
                        rel_factor=0.5,
                        default_value=0.0,
                        verbose=1):
    """
    Perform a rigid registration between a ref and moving image.
    Using ITK procedures.

    inputs:
       - ref_scan: numpy 3D image containing the reference image
       - mov_scan: numpy 3D image containing the moving image
       - ret_type: 'affine', 'rigid'
       - interpolation: ITK tranformation type (BSSpline, Linear, ...)
       - number_bins: number of histogram bins (50)
       - levels: number of levels used (3)
       - steps: Steps per level (50)
       - sampling: (0.5)
       - learning_rate: (1.0)
       - min_step: (0.001)
       - max_step: (0.2)
       - relaxation_factor: (0.5)
       - verbose: print stuff

    outputs:
       - transf: itk trasnformation
    """

    # compute the rigid transformation
    current_transf = rigid_transformation(ref_scan,
                                          mov_scan,
                                          number_bins=number_bins,
                                          levels=levels,
                                          steps=steps,
                                          sampling=sampling,
                                          learning_rate=learning_rate,
                                          min_step=min_step,
                                          max_step=max_step,
                                          relaxation_factor=rel_factor,
                                          verbose=verbose)

    if reg_type == 'affine':
        # compute the affine transformation
        current_transf = affine_transformation(ref_scan,
                                               mov_scan,
                                               initial_tf=current_transf,
                                               number_bins=number_bins,
                                               levels=levels,
                                               steps=steps,
                                               sampling=sampling,
                                               learning_rate=learning_rate,
                                               min_step=min_step,
                                               max_step=max_step,
                                               relaxation_factor=rel_factor,
                                               verbose=verbose)

    # apply the transformation
    return reg_resample(ref_scan,
                        mov_scan,
                        current_transf,
                        default_value=default_value,
                        interpolation=interpolation)


def reg_resample(ref_scan,
                 mov_scan,
                 transform,
                 default_value=0.0,
                 interpolation=sitk.sitkBSpline):
    """
    Given a computed transformation, resample two images.

    Inputs:
      - ref_scan: ref image
      - mov_scan: moving image to resample
      - tranform: rigid, affine or deformable itk transformation
      - interpolation: ITK tranformation type (BSSpline, Linear, ...)

    Outputs:
      - resampled moving scan
    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    resampled = sitk.Resample(mov, ref, transform, interpolation, default_value)
    return sitk.GetArrayFromImage(resampled)


def print_current(reg_method, tf_):
    """
    Print information about the current registration iteration

    """
    print('\t  MI (%d): %f\n\t  %s: [%s]' % (
        reg_method.GetOptimizerIteration(),
        reg_method.GetMetricValue(),
        tf.GetName(),
        ', '.join(['%s' % p for p in tf_.GetParameters()])))


def deformation_field(
        ref_scan,
        mov_scan,
        write_res=True,
        steps=50,
        sigma=1.0,
        verbose=1):
    """
    Compute the deformation field between a ref and a moving scan.

    Inputs:
      - ref_scan: reference scan
      - mov_scan: moving scan
      - def_name: deformation field name to save
      - write_res: write results to disk (res)
        steps: number of steps (50)
        sigma: (1)
        verbose)

    outputs:
     - deformation field tranformation


    """

    # convert np arrays into itk image objects
    ref = sitk.GetImageFromArray(ref_scan.astype('float32'))
    mov = sitk.GetImageFromArray(mov_scan.astype('float32'))

    if verbose > 1:
        print('\t  Deformation: ', def_name)

    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(steps)
    demons.SetStandardDeviations(sigma)

    if verbose > 1:
        demons.AddCommand(
            sitk.sitkIterationEvent,
            lambda: print('\t  Demons %d: %f' % (demons.GetElapsedIterations(),
                                                 demons.GetMetric())))

    # compute deformation fields
    def_field = demons.Execute(ref, mov)

    # if write_res:
    #    sitk.WriteImage(deformation_field, def_name)

    return def_field
    #return sitk.GetArrayFromImage(deformation_field)


def deformable_registration(ref_scan,
                            mov_scan,
                            write_res=False,
                            steps=50,
                            sigma=1.0,
                            verbose=1):
    """
    Compute the deformation field between a ref and a moving scan.

    Inputs:
      - ref_scan: reference scan
      - mov_scan: moving scan
      - def_name: deformation field name to save
      - write_res: write results to disk (res)
        steps: number of steps (50)
        sigma: (1)
        verbose)

    outputs:
     - registered_moving image
     - 3 deformation fields
    """

    # compute the deformation field
    def_field = deformation_field(
        ref_scan=ref_scan,
        mov_scan=mov_scan,
        write_res=write_res,
        steps=steps,
        sigma=sigma,
        verbose=verbose)

    # transform the def_field
    out_def = sitk.DisplacementFieldTransform(def_field)

    # resample image
    def_image = reg_resample(ref_scan, mov_scan, out_def)

    # convert the maps back
    def_maps = sitk.GetArrayFromImage(def_field)

    return def_image, def_maps

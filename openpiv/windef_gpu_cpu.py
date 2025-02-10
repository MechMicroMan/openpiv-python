"""
Created on Fri Oct  4 14:04:04 2019

@author: Theo
@modified: Alex, Erich
"""

import pathlib
from typing import Optional, Tuple, Union
import numpy as np
from skimage.util import invert

from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

from openpiv.tools import Multiprocesser, display_vector_field, transform_coordinates
from openpiv import validation, filters, tools, scaling, preprocess
from openpiv.pyprocess_gpu import extended_search_area_piv, get_rect_coordinates, \
    get_field_shape

from datetime import datetime

import cupy as cp
import cupyx.scipy.ndimage as scn

from openpiv.settings import PIVSettings

def prepare_images(
    file_a: pathlib.Path,
    file_b: pathlib.Path,
    settings: "PIVSettings",
    )-> Tuple[np.ndarray, np.ndarray]:
    """ prepares two images for the PIV pass

    Args:
        file_a (pathlib.Path): filename of frame A
        file_b (pathlib.Path): filename of frame B
        settings (_type_): windef.Settings() 
    """


    # print(f'Inside prepare_images {file_a}, {file_b}')

        # read images into numpy arrays
    frame_a = tools.imread(file_a)
    frame_b = tools.imread(file_b)
    # print(frame_a.shape)


    # Crop width if necesssary
    if frame_b.shape[1] > frame_a.shape[1]:
        offset = (frame_b.shape[1] -frame_a.shape[1]) // 2
        frame_b = frame_b[:, offset : offset + frame_a.shape[1]]

    # Crop height if necessary
    if frame_b.shape[0] > frame_a.shape[0]:
        offset = (frame_b.shape[0] - frame_a.shape[0]) // 2
        frame_b = frame_b[offset : offset + frame_a.shape[0], :]

    # Pad if necessary
    if (frame_b.shape[0] < frame_a.shape[0]) or (frame_b.shape[1] < frame_a.shape[1]):
        a = -(frame_b.shape[0] - frame_a.shape[0]) // 2
        aa = - a + frame_a.shape[0] - frame_b.shape[0]

        b = -(frame_b.shape[1] - frame_a.shape[1]) // 2
        bb = - b + frame_a.shape[1] - frame_b.shape[1]

        frame_b = np.pad(frame_b, pad_width=((a, aa), (b, bb)), mode='constant')

    if (frame_b.shape[0] != frame_a.shape[0]) or (frame_b.shape[1] != frame_a.shape[1]):
        raise ValueError('Images are different sizes.')

    # crop to roi
    if settings.roi == "full":
        pass
    else:
        frame_a = frame_a[
            settings.roi[0]:settings.roi[1],
            settings.roi[2]:settings.roi[3]
        ]
        frame_b = frame_b[
            settings.roi[0]:settings.roi[1],
            settings.roi[2]:settings.roi[3]
        ]

    return frame_a, frame_b


def piv(settings):
    """ the func fuction is the "frame" in which the PIV evaluation is done """

    # if teh settings.save_path is a string convert it to the Path
    settings.filepath_images = pathlib.Path(settings.filepath_images) 
    settings.save_directory = pathlib.Path(settings.save_directory)
    # "Below is code to read files and create a folder to store the results"
    #save_path_string = \
    #    f"OpenPIV_results_{settings.windowsizes[settings.num_iterations-1]}_{settings.save_folder_suffix}"

    if not settings.save_directory.exists():
        # os.makedirs(save_path)
        settings.save_directory.mkdir(parents=True, exist_ok=True)
        
    save_path = settings.save_directory / settings.save_filename
    
    settings.save_path = save_path

    task = Multiprocesser(
        data_dir=settings.filepath_images,
        pattern_a=settings.frame_pattern_a,
        pattern_b=settings.frame_pattern_b,
    )
    task.run(func=multipass, n_cpus=settings.num_cpus, settings=settings)


def multipass(args, settings):
    """A function to process each image pair."""

    # this line is REQUIRED for multiprocessing to work
    # always use it in your custom function

    file_a, file_b, counter = args

    # print(f'Inside func {file_a}, {file_b}, {counter}')

    # frame_a, frame_b are masked as black where we do not 
    # want to get vectors. later piv would mark it as completely black
    # and set s2n to invalid
    frame_a, frame_b = prepare_images(
        file_a,
        file_b,
        settings,
    )

    mempool = cp.get_default_memory_pool()
    cp.fft.config.get_plan_cache().set_size(0)

    frame_a = cp.array(frame_a)
    frame_b = cp.array(frame_b)

    now = datetime.now()
    print(f' {now.strftime("%H:%M:%S")}: starting pass 1')

    # "first pass"
    x, y, u, v, s2n = first_pass(
        frame_a,
        frame_b,
        settings
    )
    mempool.free_all_blocks()

    grid_mask = np.zeros_like(u, dtype=bool)

    # mask the velocity
    # u = np.ma.masked_array(u, mask=grid_mask)
    # v = np.ma.masked_array(v, mask=grid_mask)


    # validation also masks the u,v and returns another flags
    # the question is whether to merge the two masks or just keep for the 
    # reference
    if settings.validation_first_pass:
        flags = validation.typical_validation(u, v, s2n, settings)
    else:
        flags = np.zeros_like(u, dtype=bool)
    
    # "filter to replace the values that where marked by the validation"
    if (settings.num_iterations == 1 and settings.replace_vectors) \
        or (settings.num_iterations > 1):
        # for multi-pass we cannot have holes in the data
        # after the first pass
        u, v = filters.replace_outliers(
            u,
            v,
            flags,
            method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )


    # Multi pass
    for i in range(1, settings.num_iterations):

        time_diff = datetime.now() - now
        now = datetime.now()
        print(f' {now.strftime("%H:%M:%S")}: starting pass {i+1}')

        x, y, u, v, grid_mask, flags = multipass_img_deform(
            frame_a,
            frame_b,
            i,
            x,
            y,
            u,
            v,
            settings,
        )
        mempool.free_all_blocks()


    time_diff = datetime.now() - now
    now = datetime.now()
    print(f' {now.strftime("%H:%M:%S")}: completed pass {i+1}')


    # we now use only 0s instead of the image
    # masked regions.
    # we could do Nan, not sure what is best
    # u = u.filled(0.)
    # v = v.filled(0.)

    # u = cp.ma.masked_array(u, np.ma.nomask)
    # v = cp.ma.masked_array(v, np.ma.nomask)

    # pixel / frame -> pixel / second
    u /= settings.dt 
    v /= settings.dt
    
    # "scales the results pixel-> meter"
    # x, y, u, v = scaling.uniform(x, y, u, v,
    #                                 scaling_factor=settings.scaling_factor)

    # before saving we conver to the "physically relevant"
    # right-hand coordinate system with 0,0 at the bottom left
    # x to the right, y upwards
    # and so u,v
    x, y, u, v = transform_coordinates(x, y, u, v)

    # Saving
    txt_file = settings.save_path   
    print(f'Saving to {txt_file}')
    tools.save(txt_file, x, y, u, v, flags, grid_mask, fmt=settings.fmt)

    print(f"Image Pair {counter + 1}")
    print(file_a.stem, file_b.stem)


def create_deformation_field(frame, x, y, u, v, interpolation_order = 3):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Parameters
    ----------
    frame : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of
        the first frame.

    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    interpolation_order: scalar
        the degree of the interpolation of the B-splines over the rectangular mesh

    Returns
    -------
        x,y : new grid (after meshgrid)
        u,v : deformation field
    """
    y1 = y[:, 0]  # extract first coloumn from meshgrid
    x1 = x[0, :]  # extract first row from meshgrid
    side_x = np.arange(frame.shape[1])  # extract the image grid
    side_y = np.arange(frame.shape[0])

    # interpolating displacements onto a new meshgrid
    ip = RectBivariateSpline(y1, x1, u, kx=interpolation_order, ky=interpolation_order)
    ut = ip(side_y, side_x)
    # the way how to use the interpolation functions differs from matlab

    ip2 = RectBivariateSpline(y1, x1, v, kx=interpolation_order, ky=interpolation_order)
    vt = ip2(side_y, side_x)

    np.savez(f'interp_data_{x1.shape[0]}', x=x, y=y, x1=x1, y1=y1, side_y=side_y, side_x=side_x, u=u, v=v)

    x, y = np.meshgrid(side_x, side_y)

    # plt.figure()
    # plt.quiver(x1,y1,u,-v,color='r')
    # plt.quiver(x,y,ut,-vt)
    # plt.gca().invert_yaxis()
    # plt.show()

    return x, y, ut, vt


def deform_windows(frame, x, y, u, v, interpolation_order=1, interpolation_order2=3,
                   debugging=False):
    """
    Deform an image by window deformation where a new grid is defined based
    on the grid and displacements of the previous pass and pixel values are
    interpolated onto the new grid.

    Parameters
    ----------
    frame : 2d np.ndarray, dtype=np.int32
        an two dimensions array of integers containing grey levels of
        the first frame.

    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.

    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.

    u : 2d np.ndarray
        a two dimensional array containing the u velocity component,
        in pixels/seconds.

    v : 2d np.ndarray
        a two dimensional array containing the v velocity component,
        in pixels/seconds.

    interpolation_order: scalar
        the degree of the frame interpolation (deformation) of the image

    interpolation_order2: scalar
        the degree of the interpolation of the B-splines over the rectangular mesh

    Returns
    -------
    frame_def:
        a deformed image based on the meshgrid and displacements of the
        previous pass
    """

    # frame = frame.astype(np.float32)
    x, y, ut, vt = \
        create_deformation_field(frame,
                                 x, y, u, v,
                                 interpolation_order=interpolation_order2)
    # print('______------_____')
    # print(y.shape)
    # print(vt.shape)
    # print(np.array((y - vt, x + ut)).shape)
    frame_def = scn.map_coordinates(
        frame, cp.array((y - vt, x + ut)), order=interpolation_order, mode='nearest')

    if debugging:
        plt.figure()
        plt.quiver(x, y, ut, vt)
        plt.title('new, x,y, ut,vt')
        plt.show()

        plt.figure()
        plt.imshow(frame-frame_def)
        plt.title('new deformed image')
        plt.show()

    return frame_def


def first_pass(frame_a, frame_b, settings):
    # window_size,
    # overlap,
    # iterations,
    # correlation_method="circular",
    # normalized_correlation=False,
    # subpixel_method="gaussian",
    # do_sig2noise=False,
    # sig2noise_method="peak2peak",
    # sig2noise_mask=2,
    # settings):
    """
    First pass of the PIV evaluation.

    This function does the PIV evaluation of the first pass. It returns
    the coordinates of the interrogation window centres, the displacment
    u and v for each interrogation window as well as the mask which indicates
    wether the displacement vector was interpolated or not.


    Parameters
    ----------
    frame_a : 2d np.ndarray
        the first image

    frame_b : 2d np.ndarray
        the second image

    window_size : int
         the size of the interrogation window

    overlap : int
        the overlap of the interrogation window, typically it is window_size/2

    subpixel_method: string
        the method used for the subpixel interpolation.
        one of the following methods to estimate subpixel location of the peak:
        'centroid' [replaces default if correlation map is negative],
        'gaussian' [default if correlation map is positive],
        'parabolic'

    Returns
    -------
    x : 2d np.array
        array containg the x coordinates of the interrogation window centres

    y : 2d np.array
        array containg the y coordinates of the interrogation window centres

    u : 2d np.array
        array containing the u displacement for every interrogation window

    v : 2d np.array
        array containing the u displacement for every interrogation window
    
    s2n: 2d np.array of the signal to noise ratio

    """

    #     if do_sig2noise is False or iterations != 1:
    #         sig2noise_method = None  # this indicates to get out nans

    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=settings.windowsizes[0],
        overlap=settings.overlap[0],
        search_area_size=settings.windowsizes[0],
        width=settings.sig2noise_mask,
        subpixel_method=settings.subpixel_method,
        sig2noise_method=settings.sig2noise_method,
        correlation_method=settings.correlation_method,
        normalized_correlation=settings.normalized_correlation,
        use_vectorized = settings.use_vectorized,
        max_array_size=settings.max_array_size,
    )

    shapes = np.array(get_field_shape(frame_a.shape,
                                      settings.windowsizes[0],
                                      settings.overlap[0]))
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    s2n = s2n.reshape(shapes)

    x, y = get_rect_coordinates(frame_a.shape,
                           settings.windowsizes[0],
                           settings.overlap[0])

    return x, y, u, v, s2n


def multipass_img_deform(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    current_iteration: int,
    x_old: np.ndarray,
    y_old: np.ndarray,
    u_old: np.ndarray,
    v_old: np.ndarray,
    settings: "PIVSettings",
    # mask_coords: Union[np.ndarray, None]=None,
):
    """
        Multi pass of the PIV evaluation.

        This function does the PIV evaluation of the second and other passes.
        It returns the coordinates of the interrogation window centres,
        the displacement u, v for each interrogation window as well as
        the signal to noise ratio array (which is full of NaNs if opted out)


        Parameters
        ----------
        frame_a : 2d np.ndarray
            the first image

        frame_b : 2d np.ndarray
            the second image

        window_size : tuple of ints
            the size of the interrogation window

        overlap : tuple of ints
            the overlap of the interrogation window, e.g. window_size/2

        x_old : 2d np.ndarray
            the x coordinates of the vector field of the previous pass

        y_old : 2d np.ndarray
            the y coordinates of the vector field of the previous pass

        u_old : 2d np.ndarray
            the u displacement of the vector field of the previous pass
            in case of the image mask - u_old and v_old are MaskedArrays

        v_old : 2d np.ndarray
            the v displacement of the vector field of the previous pass

        subpixel_method: string
            the method used for the subpixel interpolation.
            one of the following methods to estimate subpixel location of the peak:
            'centroid' [replaces default if correlation map is negative],
            'gaussian' [default if correlation map is positive],
            'parabolic'

        interpolation_order : int
            the order of the spline interpolation used for the image deformation

        mask_coords : list of x,y coordinates (pixels) of the image mask,
            default is an empty list

        Returns
        -------
        x : 2d np.array
            array containg the x coordinates of the interrogation window centres

        y : 2d np.array
            array containg the y coordinates of the interrogation window centres

        u : 2d np.array
            array containing the horizontal displacement for every interrogation
            window [pixels]

        u : 2d np.array
            array containing the vertical displacement for every interrogation
            window it returns values in [pixels]

        grid_mask : 2d boolean np.array with the image mask in the x,y coordinates

        flags : 2D np.array of integers, flags marking 0 - valid, 1 - invalid vectors

        """
    # if not isinstance(u_old, cp.ma.MaskedArray):
    #     raise ValueError('Expected masked array')

    # calculate the y and y coordinates of the interrogation window centres.
    # Hence, the
    # edges must be extracted to provide the sufficient input. x_old and y_old
    # are the coordinates of the old grid. x_int and y_int are the coordinates
    # of the new grid

    window_size = settings.windowsizes[current_iteration] # integer only
    overlap = settings.overlap[current_iteration] # integer only, won't work for rectangular windows

    x, y = get_rect_coordinates(frame_a.shape, window_size, overlap)

    # The interpolation function dont like meshgrids as input.
    # plus the coordinate system for y is now from top to bottom
    # and RectBivariateSpline wants an increasing set

    # 1D arrays for the interpolation
    y_old = y_old[:, 0]
    x_old = x_old[0, :]

    y_int = y[:, 0]
    x_int = x[0, :]

    # interpolating the displacements from the old grid onto the new grid
    # y befor x because of numpy works row major
    # ip = RectBivariateSpline(y_old, x_old, np.ma.filled(u_old, 0.), 
    # print(f'y_old {type(y_old)}')
    # print(f'x_old {type(x_old)}')
    # print(f'u_old {type(u_old)}')
    # print(f'v_old {type(v_old)}')
    # u_old = u_old.get()
    # v_old = v_old.get()
    # print(f'u_old {type(u_old)}')
    # print(f'v_old {type(v_old)}')
    ip = RectBivariateSpline(y_old, x_old, u_old, 
                             kx=settings.interpolation_order, 
                             ky=settings.interpolation_order)
    u_pre = ip(y_int, x_int)

    # ip2 = RectBivariateSpline(y_old, x_old, np.ma.filled(v_old, 0.), 
    ip2 = RectBivariateSpline(y_old, x_old, v_old, 
                              kx=settings.interpolation_order, 
                              ky=settings.interpolation_order)
    v_pre = ip2(y_int, x_int)

    # @TKauefer added another method to the windowdeformation, 'symmetric'
    # splits the onto both frames, takes more effort due to additional
    # interpolation however should deliver better results

    # Image deformation has to occur in image coordinates
    # therefore we need to convert the results of the
    # previous pass which are stored in the physical units
    # and so y from the get_coordinates

    if settings.deformation_method == "symmetric":
        raise NotImplementedError('settings.deformation_method == "symmetric"')
        # this one is doing the image deformation (see above)
        x_new, y_new, ut, vt = create_deformation_field(
            frame_a, x, y, u_pre, v_pre,
            interpolation_order=settings.interpolation_order)
        frame_a = scn.map_coordinates(
            frame_a, ((y_new - vt / 2, x_new - ut / 2)),
            order=settings.interpolation_order, mode='nearest')
        frame_b = scn.map_coordinates(
            frame_b, ((y_new + vt / 2, x_new + ut / 2)),
            order=settings.interpolation_order, mode='nearest')
    elif settings.deformation_method == "second image":
        frame_b = deform_windows(
            frame_b, x, y, u_pre, -v_pre,
            interpolation_order=settings.interpolation_order,
            interpolation_order2=settings.interpolation_order)
    else:
        raise Exception("Deformation method is not valid.")
    now = datetime.now()
    print(f'\t{now.strftime("%H:%M:%S")}: deform_windows complete')


    # if do_sig2noise is True
    #     sig2noise_method = sig2noise_method
    # else:
    #     sig2noise_method = None

    # so we use here default circular not normalized correlation:
    # if we did not want to validate every step, remove the method
    # and save some time on cross-correlations
    if settings.sig2noise_validate is False:
        settings.sig2noise_method = None

    u, v, s2n = extended_search_area_piv(
        frame_a,
        frame_b,
        window_size=window_size,
        overlap=overlap,
        width=settings.sig2noise_mask,
        subpixel_method=settings.subpixel_method,
        sig2noise_method=settings.sig2noise_method,
        correlation_method=settings.correlation_method,
        normalized_correlation=settings.normalized_correlation,
        use_vectorized = settings.use_vectorized,
        max_array_size=settings.max_array_size,
    )

    frame_b = None
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    # get_field_shape expects tuples for rectangular windows
    shapes = np.array(get_field_shape(frame_a.shape,
                                      window_size,
                                      overlap)
                                      )
    u = u.reshape(shapes)
    v = v.reshape(shapes)
    s2n = s2n.reshape(shapes)

    u += u_pre
    v += v_pre

    grid_mask = np.zeros_like(u, dtype=bool)

    # u = np.ma.masked_array(u, mask=grid_mask)
    # v = np.ma.masked_array(v, mask=grid_mask)

    # validate in the multi-pass by default
    flags = validation.typical_validation(u, v, s2n, settings)

    now = datetime.now()
    print(f'\t{now.strftime("%H:%M:%S")}: typical_validation complete')

    if np.all(flags):
        raise ValueError("Something happened in the validation")

   
    ## Turn off remove_outliers for the last step
    if current_iteration +1 != settings.num_iterations:
        now = datetime.now()
        print(f'\t{now.strftime("%H:%M:%S")}: replace_outliers')

        # we have to replace outliers
        u, v = filters.replace_outliers(
            u,
            v,
            flags,
            method=settings.filter_method,
            max_iter=settings.max_filter_iteration,
            kernel_size=settings.filter_kernel_size,
        )
        flags = np.zeros(u.shape)

    return x, y, u, v, grid_mask, flags

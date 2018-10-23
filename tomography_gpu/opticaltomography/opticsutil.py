"""
Implement utilities using GPU

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
November 22, 2017
"""

import numpy as np
import arrayfire as af
import skimage.io as skio
import contexttimer
import tkinter
import matplotlib.pyplot as plt
from math import factorial
from scipy.ndimage.filters import uniform_filter
from tkinter.filedialog import askdirectory
from matplotlib.widgets import Slider
from os import listdir
from opticaltomography import settings

np_float_datatype   = settings.np_float_datatype
af_complex_datatype = settings.af_complex_datatype
MAX_DIM = 512*512*512 if settings.bit == 32 else 512*512*256

def show3DStack(image_3d, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1)):
    if axis == 0:
        image  = lambda index: image_3d[index, :, :]  
    elif axis == 1:
        image  = lambda index: image_3d[:, index, :]  
    else:
        image  = lambda index: image_3d[:, :, index]  
    
    current_idx= 0
    _, ax      = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig        = ax.imshow(image(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax.set_title("layer: " + str(current_idx))
    plt.axis('off')
    ax_slider  = plt.axes([0.15, 0.1, 0.65, 0.03])
    slider_obj = Slider(ax_slider, "layer", 0, image_3d.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index 
        ax.set_title("layer: " + str(index))
        fig.set_data(image(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < image_3d.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def compare3DStack(stack_1, stack_2, axis = 2, cmap = "gray", clim = (0, 1), extent = (0, 1, 0, 1)):
    assert stack_1.shape == stack_2.shape, "shape of two input stacks should be the same!"

    if axis == 0:
        image_1  = lambda index: stack_1[index, :, :]
        image_2  = lambda index: stack_2[index, :, :] 
    elif axis == 1:
        image_1  = lambda index: stack_1[:, index, :]
        image_2  = lambda index: stack_2[:, index, :] 
    else:
        image_1  = lambda index: stack_1[:, :, index]
        image_2  = lambda index: stack_2[:, :, index]

    current_idx  = 0
    _, ax        = plt.subplots(1, 2, figsize=(10, 5), sharex = 'all', sharey = 'all')
    plt.subplots_adjust(left=0.15, bottom=0.15)
    fig_1        = ax[0].imshow(image_1(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[0].axis("off")
    ax[0].set_title("stack 1, layer: " + str(current_idx))
    fig_2        = ax[1].imshow(image_2(current_idx), cmap = cmap,  clim = clim, extent = extent)
    ax[1].axis("off")
    ax[1].set_title("stack 2, layer: " + str(current_idx))
    ax_slider    = plt.axes([0.10, 0.05, 0.65, 0.03])
    slider_obj   = Slider(ax_slider, 'layer', 0, stack_1.shape[axis]-1, valinit=current_idx, valfmt='%d')
    def update_image(index):
        global current_idx
        index       = int(index)
        current_idx = index
        ax[0].set_title("stack 1, layer: " + str(index))
        fig_1.set_data(image_1(index))
        ax[1].set_title("stack 2, layer: " + str(index))
        fig_2.set_data(image_2(index))
    def arrow_key(event):
        global current_idx
        if event.key == "left":
            if current_idx-1 >=0:
                current_idx -= 1
        elif event.key == "right":
            if current_idx+1 < stack_1.shape[axis]:
                current_idx += 1
        slider_obj.set_val(current_idx)
    slider_obj.on_changed(update_image)
    plt.gcf().canvas.mpl_connect("key_release_event", arrow_key)
    plt.show()
    return slider_obj

def downSample(image, downsample_factor):
    if downsample_factor == 1:
        image_down_samp = image
    else:
        shape_ds_y      = (image.shape[0]/downsample_factor)        
        shape_ds_x      = (image.shape[1]/downsample_factor)
        slice_y         = slice(int(image.shape[0]//2-shape_ds_y//2), int(image.shape[0]//2+shape_ds_y//2))
        slice_x         = slice(int(image.shape[1]//2-shape_ds_x//2), int(image.shape[1]//2+shape_ds_x//2))
        image           = np.fft.fftshift(np.fft.fft2(image))
        image_down_samp = np.fft.ifft2(np.fft.ifftshift(image[slice_y, slice_x])).real
        image_down_samp[image_down_samp<0] = 0.0

    return image_down_samp/downsample_factor**2

def removeHalo(images, coherent_length, pixel_size, max_min = 0):
    image_num    = images.shape[2] if len(images.shape)==3 else 1
    if type(images).__module__ == np.__name__:
        images   = af.to_array(images)
    shape        = images.shape[:2]
    fxlin        = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin        = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin        = af.tile(fxlin.T, shape[0], 1)
    fylin        = af.tile(fylin, 1, shape[1])
    for filter_idx in range(3):
        if filter_idx == 0:
            filter_current       = af.Array(af.abs(fxlin))
        elif filter_idx == 1:
            filter_current       = af.Array(1.0/np.sqrt(2)*af.abs(fxlin + fylin))
        elif filter_idx == 2:
            filter_current       = af.Array(af.abs(fylin))
        filter_current[filter_current >= 1.0/coherent_length] = 1.0/coherent_length
        if filter_idx == 0:
            filters              = filter_current
        else:
            filters              = af.join(2, filters, filter_current)
    filters[:, :, :]        /= 1.0/coherent_length

    def filterHalo(image):
        mean_RI         = af.mean(image)
        filtered_images = af.ifft2(filters*af.tile(af.fft2(image), 1, 1, filters.shape[2]))
        filtered_images = af.join(2, af.real(filtered_images) + mean_RI, image)
        if max_min == 0:
            return af.imax(filtered_images, dim = 2)[0]
        else:
            return af.imin(filtered_images, dim = 2)[0]

    for image_idx in range(image_num):
        if image_idx == 0:
            halo_free_images = filterHalo(images[:, :, image_idx])
        else:
            halo_free_images = af.join(2, halo_free_images, filterHalo(images[:, :, image_idx]))

    return np.array(halo_free_images)

def backgroundNormalization(intensities, backgrounds, background_threshold, read_out_noise = 120, downsample_factor = 1):
    amplitudes = []
    assert intensities.shape == backgrounds.shape, "number of data does not match with number of background!"
    background_mean        = np.mean(backgrounds, axis = (0, 1))
    illumination_used      = np.argwhere(background_mean > background_threshold).ravel()
    background_illu        = np.empty_like(backgrounds[:, :, 0]).astype(np_float_datatype)
    for illumination_idx in illumination_used:
        intensity_illu     = intensities[:, :, illumination_idx] - read_out_noise
        intensity_illu[intensity_illu<0]   = 0.0
        uniform_filter(backgrounds[:, :, illumination_idx], size = 100, output = background_illu)
        background_illu   -= read_out_noise
        background_illu[background_illu<0] = 0.0
        intensity_down_samp= downSample(intensity_illu/background_illu, downsample_factor)
        amplitudes.append(intensity_down_samp**0.5)

    return np.asarray(amplitudes).astype(np_float_datatype).transpose(1, 2, 0), illumination_used

def loadTifffromDirectory(directory = None):
    image_stack = []
    if directory is None:
        directory   = askdirectory(parent = tkinter.Tk(), title = 'Please select a directory') + "/"
        print("load images from " + directory)
    for image_file_name in listdir(directory):
        if image_file_name.endswith(".tif") or image_file_name.endswith(".tiff"):
            image_stack.append(skio.imread(directory + image_file_name))
    if not image_stack:
        print("no image has been loaded, please check the directory or data format!")
    return np.asarray(image_stack).astype(np_float_datatype).transpose(1, 2, 0)

def calculateNumericalGradient(func, x, point, delta = 1e-4):
    function_value_0  = func(x)
    x_shift           = x.copy()
    x_shift[point]   += delta
    function_value_re = func(x_shift)
    grad_re           = (function_value_re - function_value_0)/delta
    
    x_shift           = x.copy()
    x_shift[point]   += 1.0j* delta
    function_value_im = func(x_shift)
    grad_im           = (function_value_im - function_value_0)/delta
    gradient          = 0.5*(grad_re + 1.0j*grad_im)

    return gradient

def cart2Pol(x, y):
    rho          = (x * af.conjg(x) + y * af.conjg(y))**0.5
    theta        = af.atan2(af.real(y), af.real(x)).as_type(af_complex_datatype)
    return rho, theta

def genZernikeAberration(shape, pixel_size, NA, wavelength, z_coeff = [1], z_index_list = [0]):
    assert len(z_coeff) == len(z_index_list), "number of coefficients does not match with number of zernike indices!"

    pupil             = genPupil(shape, pixel_size, NA, wavelength)
    fxlin             = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin             = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin             = af.tile(fxlin.T, shape[0], 1)
    fylin             = af.tile(fylin, 1, shape[1])
    rho, theta        = cart2Pol(fxlin, fylin)
    rho[:, :]        /= NA/wavelength

    def zernikePolynomial(z_index):
        n             = int(np.ceil((-3.0 + np.sqrt(9+8*z_index))/2.0))
        m             = 2*z_index - n*(n+2)
        zernike_poly  = af.constant(0.0, shape[0], shape[1], dtype = af_complex_datatype)
        for k in range((n-abs(m))//2+1):
            zernike_poly[:, :]  += ((-1)**k * factorial(n-k))/ \
                                    (factorial(k)*factorial(0.5*(n+m)-k)*factorial(0.5*(n-m)-k))\
                                    * rho**(n-2*k)
        return zernike_poly, m, n

    for z_coeff_index, z_index in enumerate(z_index_list):
        zernike_poly, m, _ = zernikePolynomial(z_index)
        azimuthal_function = af.sin(abs(m)*theta) if m < 0 else af.cos(abs(m)*theta)
        if z_coeff_index == 0:
            zernike_aberration = z_coeff[z_coeff_index]* zernike_poly * azimuthal_function
        else:
            zernike_aberration[:, :] += z_coeff[z_coeff_index]* zernike_poly * azimuthal_function
    
    return zernike_aberration * pupil 

def genPupil(shape, pixel_size, NA, wavelength):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin        = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin        = af.tile(fxlin.T, shape[0], 1)
    fylin        = af.tile(fylin, 1, shape[1])

    pupil_radius = NA/wavelength
    pupil        = (fxlin**2 + fylin**2 <= pupil_radius**2).as_type(af_complex_datatype)
    return pupil

def propKernel(shape, pixel_size, wavelength, prop_distance, NA = None, RI = 1.0, band_limited=True):
    assert len(shape) == 2, "pupil should be two dimensional!"
    fxlin        = genGrid(shape[1], 1/pixel_size/shape[1], flag_shift = True)
    fylin        = genGrid(shape[0], 1/pixel_size/shape[0], flag_shift = True)
    fxlin        = af.tile(fxlin.T, shape[0], 1)
    fylin        = af.tile(fylin, 1, shape[1])

    if band_limited:
        assert NA is not None, "need to provide numerical aperture of the system!"
        Pcrop    = genPupil(shape, pixel_size, NA, wavelength)
    else:
        Pcrop    = 1.0

    prop_kernel  = Pcrop * af.exp(1.0j * 2.0 * np.pi * abs(prop_distance) * Pcrop *\
                                  ((RI/wavelength)**2 - fxlin**2 - fylin**2)**0.5)
    prop_kernel  = af.conjg(prop_kernel) if prop_distance < 0 else prop_kernel
    return prop_kernel

def genGrid(size, dx, flag_shift = False):
    """
    This function generates 1D Fourier grid, and is centered at the middle of the array
    Inputs:
        size    - length of the array
        dx      - pixel size
    Optional parameters:
        flag_shift - flag indicating whether the final array is circularly shifted
                     should be false when computing real space coordinates
                     should be true when computing Fourier coordinates
    Outputs:
        xlin       - 1D Fourier grid

    """
    xlin = (af.range(size) - size//2) * dx
    if flag_shift:
        xlin = af.shift(xlin, -1 * size//2)
    return xlin.as_type(af_complex_datatype)

class ImageRotation:
    """
    A rotation class compute 3D rotation using FFT
    """
    def __init__(self, shape, axis = 0, pad = False, pad_value = 0, flag_gpu_inout=False, flag_inplace = False):
        self.dim       = np.asarray(shape)
        self.axis      = axis
        self.pad_value = pad_value
        self.flag_gpu_inout = flag_gpu_inout
        self.flag_inplace = flag_inplace
        if pad:
                self.pad_size       = np.ceil(self.dim / 2.0).astype('int')
                self.pad_size[self.axis] = 0
                self.dim           += 2*self.pad_size
        else:
            self.pad_size  = np.asarray([0,0,0])

        self.dim          = [int(size) for size in self.dim]
        self.range_crop_x = slice(self.pad_size[1],self.pad_size[1] + shape[1])
        self.range_crop_y = slice(self.pad_size[0],self.pad_size[0] + shape[0])
        self.range_crop_z = slice(self.pad_size[2],self.pad_size[2] + shape[2])

        self.x            = af.moddims(af.range(self.dim[1]) - self.dim[1]/2, 1, self.dim[1], 1)
        self.y            = af.moddims(af.range(self.dim[0]) - self.dim[0]/2, self.dim[0], 1, 1)
        self.z            = af.moddims(af.range(self.dim[2]) - self.dim[2]/2, 1, 1, self.dim[2])

        self.kx           = af.moddims((1.0/self.dim[1]) * \
                            af.shift(af.range(self.dim[1]) - self.dim[1]/2, self.dim[1]//2)\
                            , 1, self.dim[1], 1)
        self.ky           = af.moddims((1.0/self.dim[0]) * \
                            af.shift(af.range(self.dim[0]) - self.dim[0]/2, self.dim[0]//2)\
                            , self.dim[0], 1, 1)
        self.kz           = af.moddims((1.0/self.dim[2]) * \
                            af.shift(af.range(self.dim[2]) - self.dim[2]/2, self.dim[2]//2)\
                            , 1, 1, self.dim[2])
        
        #Compute FFTs sequentially if object size is too large
        self.slice_per_tile = int(np.min([np.floor(MAX_DIM * self.dim[self.axis] / np.prod(self.dim)), self.dim[self.axis]]))

    def rotate(self, obj, theta):
        if theta == 0:
            return obj
        else:
            for idx_start in range(0, obj.shape[self.axis], self.slice_per_tile):
                idx_end = np.min([obj.shape[self.axis], idx_start+self.slice_per_tile])
                idx_slice = slice(idx_start, idx_end)
                self.dim[self.axis] = int(idx_end - idx_start)
                if self.axis == 0:
                    self.range_crop_y = slice(0, self.dim[self.axis])
                    obj[idx_slice,:,:] = self._rotate3D(obj[idx_slice,:,:], theta, self.pad_value)
                elif self.axis == 1:
                    self.range_crop_x = slice(0, self.dim[self.axis])
                    obj[:,idx_slice,:] = self._rotate3D(obj[:,idx_slice,:], theta, self.pad_value)
                elif self.axis == 2:
                    self.range_crop_z = slice(0, self.dim[self.axis])
                    obj[:,:,idx_slice] = self._rotate3D(obj[:,:,idx_slice], theta, self.pad_value)
            self.dim[self.axis] = obj.shape[self.axis]
            return obj

    def rotate_adj(self, obj, theta):
        if theta == 0:
            return obj
        else:
            for idx_start in range(0, obj.shape[self.axis], self.slice_per_tile):
                idx_end = np.min([obj.shape[self.axis], idx_start+self.slice_per_tile])
                idx_slice = slice(idx_start, idx_end)
                self.dim[self.axis] = int(idx_end - idx_start)
                if self.axis == 0:
                    self.range_crop_y = slice(0, self.dim[self.axis])
                    obj[idx_slice,:,:] = self._rotate3D(obj[idx_slice,:,:], -1*theta, 0)
                elif self.axis == 1:
                    self.range_crop_x = slice(0, self.dim[self.axis])
                    obj[:,idx_slice,:] = self._rotate3D(obj[:,idx_slice,:], -1*theta, 0)
                elif self.axis == 2:
                    self.range_crop_z = slice(0, self.dim[self.axis])
                    obj[:,:,idx_slice] = self._rotate3D(obj[:,:,idx_slice], -1*theta, 0)
            self.dim[self.axis] = obj.shape[self.axis]
            return obj            

    def _shearInplace(self, obj, phase_shift):
        af.fft_inplace(obj)
        obj *= phase_shift
        af.ifft_inplace(obj)
        return obj
    
    def _rotate3D(self, obj, theta, pad_value):
        """
        This function rotates a 3D image by shearing, (applied in Fourier space)
        ** Note: the rotation is performed along the z axis

        [ cos(theta)  -sin(theta) ] = [ 1  alpha ] * [ 1     0  ] * [ 1  alpha ]
        [ sin(theta)  cos(theta)  ]   [ 0    1   ]   [ beta  1  ]   [ 0    1   ]
        alpha = tan(theta/2)
        beta = -sin(theta)

        Shearing in one shapeension is applying phase shift in 1D fourier transform
        Input:
          obj: 3D array (supposed to be an image), the axes are [z,y,x]
          theta: desired angle of rotation in *degrees*
        Output:
          obj_rotate: rotate 3D array
        """
        with contexttimer.Timer() as timer:
            theta      *= np.pi / 180.0
            alpha       = 1.0 * np.tan(theta / 2.0)
            beta        = np.sin(-1.0 * theta)
            obj_rotate = af.constant(pad_value, self.dim[0], self.dim[1], self.dim[2], dtype = af_complex_datatype)
            if self.flag_gpu_inout:
                obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z] = obj
            else:
                obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z] = af.to_array(obj)


            if self.axis == 0:
                x            = af.tile(self.x, self.dim[2], 1, self.dim[0])
                z            = af.tile(self.z, self.dim[1], self.dim[0], 1)
                kx           = af.tile(af.reorder(self.kx, 1, 0, 2), 1, self.dim[0], self.dim[2])
                kz           = af.tile(af.reorder(self.kz, 2, 1, 0), 1, self.dim[1], self.dim[0])
                z_phaseshift = af.exp(-2.0j * np.pi * kz * x * alpha)
                x_phaseshift = af.exp(-2.0j * np.pi * kx * z * beta)
                
                obj_rotate = af.reorder(obj_rotate, 2, 1, 0)
                obj_rotate = self._shearInplace(obj_rotate, z_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 1, 2, 0)
                obj_rotate = self._shearInplace(obj_rotate, x_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 2, 0, 1)
                obj_rotate = self._shearInplace(obj_rotate, z_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 2, 1, 0)
                
            elif self.axis == 1:
                y            = af.tile(af.reorder(self.y, 2, 0, 1), self.dim[2], 1, self.dim[1])
                z            = af.tile(self.z, self.dim[0], self.dim[1], 1)
                ky           = af.tile(self.ky, 1, self.dim[1], self.dim[2])
                kz           = af.tile(af.reorder(self.kz, 2, 0, 1), 1, self.dim[0], self.dim[1])
                
                z_phaseshift = af.exp(-2.0j * np.pi * kz * y * alpha)
                y_phaseshift = af.exp(-2.0j * np.pi * ky * z * beta)
                
                obj_rotate = af.reorder(obj_rotate, 2, 0, 1)
                obj_rotate = self._shearInplace(obj_rotate, z_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 1, 2, 0)
                obj_rotate = self._shearInplace(obj_rotate, y_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 2, 0, 1)
                obj_rotate = self._shearInplace(obj_rotate, z_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 1, 2, 0)

            elif self.axis == 2:
                x            = af.tile(self.x, self.dim[0], 1, self.dim[2])
                y            = af.tile(af.reorder(self.y, 1, 2, 0), self.dim[1], self.dim[2], 1)
                kx           = af.tile(af.reorder(self.kx, 1, 2, 0), 1, self.dim[2], self.dim[0])
                ky           = af.tile(self.ky, 1, self.dim[1], self.dim[2])

                x_phaseshift = af.exp(-2.0j * np.pi * kx * y * alpha)
                y_phaseshift = af.exp(-2.0j * np.pi * ky * x * beta)

                obj_rotate = af.reorder(obj_rotate, 1, 2, 0)
                obj_rotate = self._shearInplace(obj_rotate, x_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 2, 0, 1)
                obj_rotate = self._shearInplace(obj_rotate, y_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 1, 2, 0)
                obj_rotate = self._shearInplace(obj_rotate, x_phaseshift)
                obj_rotate = af.reorder(obj_rotate, 2, 0, 1)
            if self.flag_inplace:
                if self.flag_gpu_inout:
                    obj[:] = obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z]
                    return obj
                else:
                    obj[:] = np.array(obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z])        
                    return obj
            if self.flag_gpu_inout:
                return obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z]
            else:
                obj_rotate = np.array(obj_rotate[self.range_crop_y, self.range_crop_x, self.range_crop_z])        
                return obj_rotate
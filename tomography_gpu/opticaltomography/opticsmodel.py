"""
Implement optics algorithms for optical phase tomography using GPU

Michael Chen   mchen0405@berkeley.edu
David Ren      david.ren@berkeley.edu
November 22, 2017
"""

import contexttimer

import numpy as np
import arrayfire as af
from opticaltomography.opticsutil import genGrid, genPupil, propKernel, genZernikeAberration
from opticaltomography import settings

np_complex_datatype = settings.np_complex_datatype
af_complex_datatype = settings.af_complex_datatype

class Aperture:
    """
    Class for optical aperture (general)
    """
    def __init__(self, shape, pixel_size, na, pad = True, pad_size = None, **kwargs):
        """
        shape:          shape of object (y,x,z)
        pixel_size:     pixel size of the system
        na:             NA of the system
        pad:            boolean variable to pad the reconstruction
        pad_size:       if pad is true, default pad_size is shape//2. Takes a tuple, pad size in dimensions (y, x)
        """
        self.shape          = shape
        self.pixel_size     = pixel_size
        self.na             = na
        self.pad            = pad
        if self.pad:
            self.pad_size       = pad_size
            if self.pad_size == None:
                self.pad_size   = (self.shape[0]//4, self.shape[1]//4)
            self.row_crop   = slice(self.pad_size[0], self.shape[0] - self.pad_size[0])
            self.col_crop   = slice(self.pad_size[1], self.shape[1] - self.pad_size[1])
        else:
            self.row_crop   = slice(0, self.shape[0])
            self.col_crop   = slice(0, self.shape[1])

    def forward(self):
        pass

    def adjoint(self):
        pass

class Aberration(Aperture):
    """
    Aberration class used for pupil recovery 
    """
    def __init__(self, shape, pixel_size, wavelength, na, pad = True, flag_update = False, pupil_step_size = 1.0, update_method = "gradient", **kwargs):
        """
        Initialization of the class

        wavelength:         wavelength of light
        flag_update:        boolean variable to update pupil
        pupil_step_size:    if update the pupil, what is a step size for gradient method
        update_method:      can be "gradient" or "GaussNewton"
        """
        super().__init__(shape, pixel_size, na, pad, **kwargs)
        self.pupil            = genPupil(self.shape, self.pixel_size, self.na, wavelength)
        self.wavelength       = wavelength
        self.pupil_support    = self.pupil.copy()
        self.pupil_step_size  = pupil_step_size
        self.flag_update      = flag_update
        self.update_method    = update_method

    def forward(self, field):
        """Apply pupil"""
        self.field_f          = af.fft2(field)
        if self.update_method == "GaussNewton":
            self.approx_hessian[:, :] += self.field_f*af.conjg(self.field_f)
        field_pupil           = af.ifft2(self.pupil * self.field_f)[self.row_crop, self.col_crop]
        return field_pupil

    def adjoint(self, field):
        """Adjoint operator for pupil (and estimate pupil if selected)"""
        field_f               = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)
        field_f[self.row_crop, self.col_crop] = field
        af.fft2_inplace(field_f)
        field_pupil_adj       = af.ifft2(af.conjg(self.pupil) * field_f)
        #Pupil recovery
        if self.flag_update:
            self.pupil_gradient[:, :] += af.conjg(self.field_f) * field_f
            self.measure_count        += 1
            if self.measure_count == self.measurement_num:
                self._update()
                self.measure_count     = 0
        return field_pupil_adj

    def _update(self):
        """function to recover pupil"""
        if self.update_method == "gradient":
            self.pupil[:, :] -= self.pupil_step_size * self.pupil_gradient * self.pupil_support
        elif self.update_method == "GaussNewton":
            self.pupil[:, :] -= self.pupil_step_size * 0.25 / (self.approx_hessian + 1e-8) * self.pupil_gradient * self.pupil_support
            self.approx_hessian[:, :] = 0.0
        else:
            print("there is no update_method \"%s\"!" %(self.update_method))
            raise
        self.pupil_gradient[:, :] = 0.0

    def getPupil(self):
        """function to retrieve pupil"""
        return self.pupil

    def setPupil(self, pupil = None):
        """Input arbitratyt pupil"""
        self.pupil[:, :]      = self.pupil_support if pupil is None else pupil

    def setZernikePupil(self, z_coeff = [1], z_index_list = [0]):
        """Set pupil according to Zernike coefficients"""
        self.pupil[:, :]      = self.pupil_support *\
                                af.exp(1.0j*genZernikeAberration(self.shape, self.pixel_size, self.na, self.wavelength, z_coeff = z_coeff, z_index_list = z_index_list))

    def setUpdateParams(self, flag_update = None, pupil_step_size = None, update_method = None, global_update = False, measurement_num = None):
        """Modify update parameters for pupil"""
        self.setPupil()
        self.flag_update      = flag_update if flag_update is not None else self.flag_update
        if self.flag_update:
            self.update_method    = update_method if update_method is not None else self.update_method
            self.pupil_step_size  = pupil_step_size if pupil_step_size is not None else self.pupil_step_size
            self.pupil_gradient   = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)
            self.measurement_num  = measurement_num if global_update else 1
            self.measure_count    = 0                
            if self.update_method == "GaussNewton":
                self.approx_hessian   = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)

class Defocus(Aperture):
    """Defocus subclass for tomography"""
    def __init__(self, shape, pixel_size, wavelength, na, RI_measure = 1.0, pad = True, **kwargs):
        """
        Initialization of the class

        RI_measure: refractive index on the detection side (example: oil immersion objectives)
        """
        super().__init__(shape, pixel_size, na, pad, **kwargs)

        fxlin                   = genGrid(self.shape[1], 1.0/self.pixel_size/self.shape[1], flag_shift = True)
        fylin                   = genGrid(self.shape[0], 1.0/self.pixel_size/self.shape[0], flag_shift = True)
        fxlin                   = af.tile(fxlin.T, self.shape[0], 1)
        fylin                   = af.tile(fylin, 1, self.shape[1])
        self.pupil              = genPupil(self.shape, self.pixel_size, self.na, wavelength)
        self.prop_kernel_phase  = 1.0j*2.0*np.pi*self.pupil*((RI_measure/wavelength)**2 - fxlin*af.conjg(fxlin) - fylin*af.conjg(fylin))**0.5

    def forward(self, field, propagation_distances):
        """defocus with angular spectrum"""
        field_defocus           = self.pupil * af.fft2(field)
        field_defocus           = af.tile(field_defocus, 1, 1, len(propagation_distances))
        for z_idx, propagation_distance in enumerate(propagation_distances):
            propagation_kernel          = af.exp(self.prop_kernel_phase*propagation_distances[z_idx])
            field_defocus[:, :, z_idx] *= propagation_kernel
        af.ifft2_inplace(field_defocus)
        return field_defocus[self.row_crop, self.col_crop]

    def adjoint(self, residual, propagation_distances):
        """adjoint operator for defocus with angular spectrum"""
        field_focus             = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)
        field_pad               = af.constant(0.0, self.shape[0], self.shape[1], dtype = af_complex_datatype)
        for z_idx, propagation_distance in enumerate(propagation_distances):
            field_pad[self.row_crop, self.col_crop] = residual[:, :, z_idx]
            propagation_kernel_conj  = af.exp(-1.0*self.prop_kernel_phase*propagation_distances[z_idx])
            field_focus[:, :]       += propagation_kernel_conj * af.fft2(field_pad)
        field_focus            *= self.pupil
        af.ifft2_inplace(field_focus)
        return field_focus

class ScatteringModels:
    """
    Core of the scattering models
    """
    def __init__(self, phase_obj_3d, wavelength, slice_binning_factor = 1, pad = True, **kwargs):
        """
        Initialization of the class

        phase_obj_3d:           object of the class PhaseObject3D 
        wavelength:             wavelength of the light
        slice_binning_factor:   the object is compress in z-direction by this factor
        pad:                    boolean variable to pad the reconstruction.
                                If true, reconstruction size is padded 
                                If false, reconstruction size is the same as measurement size
        """
        self.pad                  = pad
        self.slice_binning_factor = slice_binning_factor
        self._shape_full          = phase_obj_3d.shape
        self.shape                = phase_obj_3d.shape[0:2] + (int(np.ceil(phase_obj_3d.shape[2]/self.slice_binning_factor)),)
        self.RI                   = phase_obj_3d.RI
        self.wavelength           = wavelength
        self.pixel_size           = phase_obj_3d.pixel_size
        self.pixel_size_z         = phase_obj_3d.pixel_size_z * self.slice_binning_factor
        self.back_scatter         = False
        #Broadcasts b to a, size(a) > size(b)
        self.assign_broadcast     = lambda a, b : a - a + b

    def _genRealGrid(self, fftshift = False):
        xlin                = genGrid(self.shape[1], self.pixel_size, flag_shift = fftshift)
        ylin                = genGrid(self.shape[0], self.pixel_size, flag_shift = fftshift)
        xlin                = af.tile(xlin.T, self.shape[0], 1)
        ylin                = af.tile(ylin, 1, self.shape[1])
        return xlin, ylin

    def _genFrequencyGrid(self):
        fxlin               = genGrid(self.shape[1], 1.0/self.pixel_size/self.shape[1], flag_shift = True)
        fylin               = genGrid(self.shape[0], 1.0/self.pixel_size/self.shape[0], flag_shift = True)
        fxlin               = af.tile(fxlin.T, self.shape[0], 1)
        fylin               = af.tile(fylin, 1, self.shape[1])
        return fxlin, fylin

    def _genIllumination(self, fx_illu, fy_illu):
        fx_illu, fy_illu    = self._setIlluminationOnGrid(fx_illu, fy_illu)
        xlin, ylin          = self._genRealGrid()
        fz_illu             = ((self.RI/self.wavelength)**2 - fx_illu**2 - fy_illu**2)**0.5
        illumination_xy     = af.exp(1.0j*2.0*np.pi*(fx_illu*xlin + fy_illu*ylin))
        return illumination_xy, fx_illu, fy_illu, fz_illu

    def _setIlluminationOnGrid(self, fx_illu, fy_illu):
        dfx                 = 1.0/self.pixel_size/self.shape[1]
        dfy                 = 1.0/self.pixel_size/self.shape[0]
        fx_illu_on_grid     = np.round(fx_illu/dfx)*dfx
        fy_illu_on_grid     = np.round(fy_illu/dfy)*dfy
        return fx_illu_on_grid, fy_illu_on_grid


    def _binObject(self, obj, adjoint = False):
        """
        function to bin the object by factor of slice_binning_factor
        """
        if self.slice_binning_factor == 1:
            return obj
        if adjoint:
            obj_out = af.constant(0.0, self._shape_full[0], self._shape_full[1], self._shape_full[2], dtype = af_complex_datatype)
            for idx in range((self.shape[2]-1)*self.slice_binning_factor, -1, -self.slice_binning_factor):
                idx_slice = slice(idx, np.min([obj_out.shape[2],idx+self.slice_binning_factor]))
                obj_out[:,:,idx_slice] = af.broadcast(self.assign_broadcast, obj_out[:,:,idx_slice], obj[:,:,idx//self.slice_binning_factor])
        else:
            obj_out = af.constant(0.0, self.shape[0], self.shape[1], self.shape[2], dtype = af_complex_datatype)
            for idx in range(0, obj.shape[2], self.slice_binning_factor):
                idx_slice = slice(idx, np.min([obj.shape[2], idx+self.slice_binning_factor]))
                obj_out[:,:,idx//self.slice_binning_factor] = af.sum(obj[:,:,idx_slice], 2)
        return obj_out                    
    
    def forward(self, x_obj, fx_illu, fy_illu):
        pass

    def adjoint(self, residual, cache):
        pass


class MultiTransmittance(ScatteringModels):
    """
    MultiTransmittance scattering model. This class also serves as a parent class for all multi-slice scattering methods
    """
    def __init__(self, phase_obj_3d, wavelength, **kwargs):
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        self.slice_separation              = [sum(phase_obj_3d.slice_separation[x:x+self.slice_binning_factor]) \
                                              for x in range(0,len(phase_obj_3d.slice_separation),self.slice_binning_factor)]
        sample_thickness                   = np.sum(self.slice_separation)
        total_prop_distance                = np.sum(self.slice_separation[0:self.shape[2]-1])
        self.distance_end_to_center        = total_prop_distance - sample_thickness/2.
        fxlin, fylin                       = self._genFrequencyGrid()
        fzlin                              = ((self.RI/self.wavelength)**2 - fxlin*af.conjg(fxlin) - fylin*af.conjg(fylin))**0.5
        self.prop_kernel_phase             = 1.0j * 2.0 * np.pi * fzlin

        if np.abs(np.mean(self.slice_separation) - self.pixel_size_z) < 1e-6 or self.slice_binning_factor > 1:
            self.focus_at_center           = True
            self.initial_z_position        = -1 * (self.shape[2]//2) * self.pixel_size_z
        else:
            self.focus_at_center           = False
            self.initial_z_position        = 0.0

    def forward(self, trans_obj, fx_illu, fy_illu):
        if self.slice_binning_factor > 1:
            print("Slicing is not implemented for MultiTransmittance algorithm!")
            raise
        
        #compute illumination
        field, fx_illu, fy_illu, fz_illu   = self._genIllumination(fx_illu, fy_illu)
        field[:, :]                       *= np.exp(1.0j * 2.0 * np.pi * fz_illu * self.initial_z_position)
        
        #multi-slice transmittance forward propagation
        field_layer_conj                   = af.constant(0.0, trans_obj.shape[0], trans_obj.shape[1], trans_obj.shape[2], dtype = af_complex_datatype)
        if (type(trans_obj).__module__ == np.__name__):
            trans_obj_af                   = af.to_array(trans_obj)
            flag_gpu_inout                 = False
        else:
            trans_obj_af                   = trans_obj
            flag_gpu_inout                 = True

        for layer in range(self.shape[2]):
            field_layer_conj[:, :, layer]  = af.conjg(field)
            field[:, :]                   *= trans_obj_af[:, :, layer]
            if layer < self.shape[2]-1:
                field                      = self._propagationInplace(field, self.slice_separation[layer])
        
        #store intermediate variables for adjoint operation
        cache                              = (trans_obj_af, field_layer_conj, flag_gpu_inout)

        if self.focus_at_center:
            #propagate to volume center
            field                          = self._propagationInplace(field, self.distance_end_to_center, adjoint = True)
        return {'forward_scattered_field': field, 'cache': cache}

    def adjoint(self, residual, cache):
        trans_obj_af, field_layer_conj_or_grad, flag_gpu_inout = cache
        
        #back-propagte to volume center
        field_bp                           = residual
        if self.focus_at_center:
            #propagate to the last layer
            field_bp                       = self._propagationInplace(field_bp, self.distance_end_to_center)

        #multi-slice transmittance backward
        for layer in range(self.shape[2]-1, -1, -1):
            field_layer_conj_or_grad[:, :, layer] = field_bp * field_layer_conj_or_grad[:, :, layer]
            if layer > 0:
                field_bp[:, :]            *= af.conjg(trans_obj_af[:, :, layer])
                field_bp                   = self._propagationInplace(field_bp, self.slice_separation[layer-1], adjoint = True)
        if flag_gpu_inout:
            return {'gradient':field_layer_conj_or_grad}
        else:
            return {'gradient':np.array(field_layer_conj_or_grad)}

    def _propagationInplace(self, field, propagation_distance, adjoint = False, in_real = True):
        """
        propagation operator that uses angular spectrum to propagate the wave

        field:                  input field
        propagation_distance:   distance to propagate the wave
        adjoint:                boolean variable to perform adjoint operation (i.e. opposite direction)
        """
        if in_real:
            af.fft2_inplace(field)
        if adjoint:
            field[:, :] *= af.conjg(af.exp(self.prop_kernel_phase * propagation_distance))
        else:
            field[:, :] *= af.exp(self.prop_kernel_phase * propagation_distance)
        if in_real:
            af.ifft2_inplace(field)
        return field

class MultiPhaseContrast(MultiTransmittance):
    """ MultiPhaseContrast, solves directly for the phase contrast {i.e. Transmittance = exp(sigma * PhaseContrast)} """
    def __init__(self, phase_obj_3d, wavelength, sigma = 1, **kwargs):
        super().__init__(phase_obj_3d, wavelength, **kwargs)
        self.sigma = sigma

    def forward(self, contrast_obj, fx_illu, fy_illu):
        #compute illumination
        field, fx_illu, fy_illu, fz_illu   = self._genIllumination(fx_illu, fy_illu)
        field[:, :]                       *= np.exp(1.0j * 2.0 * np.pi * fz_illu * self.initial_z_position)
        #multi-slice transmittance forward
        field_layer_conj                   = af.constant(0.0, self.shape[0], self.shape[1], self.shape[2], dtype = af_complex_datatype)
        
        if (type(contrast_obj).__module__ == np.__name__):
            phasecontrast_obj_af           = af.to_array(contrast_obj)
            flag_gpu_inout                 = False
        else:
            phasecontrast_obj_af           = contrast_obj
            flag_gpu_inout                 = True

        #Binning
        obj_af = self._binObject(phasecontrast_obj_af)

        #Potentials to Transmittance
        obj_af = af.exp(1.0j * self.sigma * obj_af)

        for layer in range(self.shape[2]):
            field_layer_conj[:, :, layer]  = af.conjg(field)
            field[:, :]                   *= obj_af[:, :, layer]
            if layer < self.shape[2]-1:
                field                      = self._propagationInplace(field, self.slice_separation[layer])
        #propagate to volume center
        cache                              = (obj_af, field_layer_conj, flag_gpu_inout)

        if self.focus_at_center:
            #propagate to volume center
            field                          = self._propagationInplace(field, self.distance_end_to_center, adjoint = True)

        return {'forward_scattered_field': field, 'cache': cache}

    def adjoint(self, residual, cache):
        phasecontrast_obj_af, field_layer_conj_or_grad, flag_gpu_inout = cache
        trans_obj_af_conj                  = af.conjg(phasecontrast_obj_af)
        #back-propagte to volume center
        field_bp                           = residual
        #propagate to the last layer
        if self.focus_at_center:
            field_bp                       = self._propagationInplace(field_bp, self.distance_end_to_center)
        #multi-slice transmittance backward
        for layer in range(self.shape[2]-1, -1, -1):
            field_layer_conj_or_grad[:, :, layer]= field_bp * field_layer_conj_or_grad[:, :, layer] * (-1.0j) * self.sigma * trans_obj_af_conj[:,:,layer]
            if layer > 0:
                field_bp[:, :]            *= trans_obj_af_conj[:, :, layer]
                field_bp                   = self._propagationInplace(field_bp, self.slice_separation[layer-1], adjoint = True)
        
        #Unbinning
        grad = self._binObject(field_layer_conj_or_grad, adjoint = True)

        if flag_gpu_inout:
            return {'gradient':grad}
        else:
            return {'gradient':np.array(grad)}

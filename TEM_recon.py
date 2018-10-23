import numpy as np
import TEM_misc as misc
from opticaltomography.opticsalg import PhaseObject3D, TomographySolver, AlgorithmConfigs

class TEM_recon_gpu():
    """
    Reconstruction class for TEM phase contrast tomography, GPU
    """    
    def __init__(self, data, opts):
        self.opts = opts
        # Load data
        self.wavelength = np.asscalar(misc.checkDataNames(data, ("lambda","wavelength")))
        self.pixel_size = np.asscalar(misc.checkDataNames(data, ("pixelSize","pixel_size")))
        self.na = np.asscalar(misc.checkDataNames(data, ("na","NA"), self.wavelength))
        self.dz = np.asscalar(data.get("dz"))
        self.sigma = np.asscalar(data.get("sigma"))
        
        # cost function
        self.cost = np.zeros(int(self.opts["maxitr"]))

        # Measurement
        self.amplitude_measure = (data.get("intensity_measure") ** 0.5)
        
        self.obj_shape = data.get("obj_shape", None)
        
        if self.obj_shape is None:
            self.xn = np.asscalar(data.get("xn"))
            self.obj_shape = (self.xn,self.xn,self.xn)
        else:
            self.obj_shape = tuple(np.asscalar(n) for n in self.obj_shape.ravel())

        if self.obj_shape[0] > self.amplitude_measure.shape[0]:
            self.pad = True
            self.pad_size = ((self.obj_shape[0] - self.amplitude_measure.shape[0])//2,\
                             (self.obj_shape[1] - self.amplitude_measure.shape[1])//2)
        else:
            self.pad = False
            self.pad_size = (0,0)

        self.tilt_angles = data.get("tilt_angles").ravel()

        self.defocus_stack = data.get("defocus_stack").ravel()

        self.slice_binning_factor = int(opts["slice_binning_factor"])
        if "flag_rotation_pad" in opts:
            self.flag_rotation_pad = bool(opts["flag_rotation_pad"])
        else:
            self.flag_rotation_pad = True

        # Initialization
        self.obj_init = data.get("init", 0)

        fx_illu_list = [0]
        fy_illu_list = [0]
        print("pad is:", self.pad)
        solver_params = dict(wavelength = self.wavelength, na = self.na, \
                             propagation_distance_list = self.defocus_stack, rotation_angle_list = self.tilt_angles, \
                             RI_measure = 1.0, sigma = self.sigma*self.dz, \
                             fx_illu_list = fx_illu_list, fy_illu_list = fy_illu_list, \
                             pad = self.pad, pad_size = self.pad_size, \
                             slice_binning_factor = self.slice_binning_factor, \
                             rotation_pad = self.flag_rotation_pad)
        
        phase_obj_3d       = PhaseObject3D(shape = self.obj_shape, \
                                     voxel_size = (self.pixel_size, self.pixel_size, self.dz))
        self.solver_obj    = TomographySolver(phase_obj_3d, **solver_params)
        self.solver_obj.setScatteringMethod(model = "MultiPhaseContrast")

    def run(self):
        configs            = AlgorithmConfigs()
        configs.batch_size = self.opts["gradient_batch_size"]
        configs.method     = self.opts["update_mode"]
        configs.random_order = self.opts["random_order"]
        configs.restart    = True
        configs.max_iter   = self.opts["maxitr"]
        configs.stepsize   = 1/(self.opts["fista_L"])
        configs.error      = []

        self.configs       = configs
        # Parsing Priors
        if self.opts["flag_reg"]:
            for reg_type in self.opts["reg_type"]:
                if reg_type == "positivity_and_real": #DEPRECATED
                    configs.pure_real = True
                    configs.positivity_real = (True, "larger")
                elif reg_type == "real":
                    configs.pure_real = True
                elif reg_type == "positivity":
                    configs.positivity_real = (True, "larger")
                elif reg_type == "tv":
                    configs.total_variation     = True
                    configs.total_variation_gpu = True
                    configs.reg_tv              = self.opts["reg_params"].get("reg_tv",      configs.reg_tv)
                    configs.max_iter_tv         = self.opts["reg_params"].get("max_iter_tv", configs.max_iter_tv)
                elif reg_type == "lasso":
                    configs.lasso               = True
                    configs.reg_lasso           = self.opts["reg_params"].get("reg_lasso",   configs.reg_lasso)
                else:
                    Print("Regularizer not recognized!")

        recon_obj_3d = self.solver_obj.solve(configs, self.amplitude_measure, x_init = self.obj_init)
        self.current_rec = recon_obj_3d
        self.cost = configs.error

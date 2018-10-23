import numpy as np

def checkDataNames(data, name_list, default = None):
    """
    This function checks multiple possible names for the variable in the dictionary
    It is assumed that only one name in the name_list is an existing field in the dict (data),
    The value corresponding to the first existing name in the name_list will be rerurned.
    In the case where no variables could be found, default value is returned:
    """
    flag_found = False
    for name in name_list:
        if name in data:
            val = data[name]
            flag_found = True
            break
    if flag_found == False:
        val = default
    return val

def generatePupil(shape, dx, NA, lamb, flag_3D = False):
    zn = shape[0]
    yn = shape[1]
    xn = shape[2]
    dkz = 1./(zn*dx)
    dky = 1./(yn*dx)
    dkx = 1./(xn*dx)
    kz = np.linspace(-zn//2, zn//2, zn, endpoint=False) * dkz
    ky = np.linspace(-yn//2, yn//2, yn, endpoint=False) * dky
    kx = np.linspace(-xn//2, xn//2, xn, endpoint=False) * dkx
    if flag_3D:
        [kzz, kyy, kxx] = np.meshgrid(kz**2,ky**2, kx**2, indexing = 'ij')
        r = np.sqrt(kxx + kyy + kzz)     
    else:
        [kxx, kyy] = np.meshgrid(kx**2,ky**2)
        r = np.sqrt(kxx + kyy)
    p = (r < NA/lamb).astype(np.complex128)                 
    return p

def apply3DPupil(x, dx, NA, lamb):
    p = generatePupil(x.shape, dx, NA, lamb, flag_3D = True)
    return np.real(np.fft.ifftn(np.fft.fftshift(np.fft.fftshift(np.fft.fftn(x)) * p)))
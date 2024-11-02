# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        Diffractio.py
# Purpose:     Class for controlling the other modules.
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


"""
This module 
    

"""
# flake8: noqa


# import copy
# import copyreg
# import multiprocessing
# import time
# import types



# from numpy import angle, array, concatenate, exp, linspace, pi, shape, sqrt, zeros
# from numpy.lib.scimath import sqrt as csqrt
# from scipy.fftpack import fft, fftshift, ifft
# from scipy.interpolate import interp1d
# from scipy.special import hankel1

# from .__init__ import degrees, mm, np, plt

# from .config import bool_raise_exception, Draw_X_Options, get_scalar_options, empty_types
# from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
# from .utils_common import get_date, load_data_common, save_data_common, check_none, add, oversampling
# from .utils_drawing import normalize_draw
# from .utils_math import (fft_filter, get_edges, nearest, reduce_to_1, Bluestein_dft_x, get_k, nearest2)
# from .utils_multiprocessing import (_pickle_method, _unpickle_method,
#                                     execute_multiprocessing)
# from .utils_optics import field_parameters, normalize_field

# copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# num_max_processors = multiprocessing.cpu_count()


from .config import Options_Diffractio_kind, Options_Diffractio_frame

from diffractio.scalar_fields_X import Scalar_field_X
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_fields_XZ import Scalar_field_XZ
from diffractio.scalar_fields_Z import Scalar_field_Z

from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_masks_XZ import Scalar_mask_XZ

from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.scalar_sources_XY import Scalar_source_XY

from diffractio.vector_fields_X import Vector_field_X
from diffractio.vector_fields_XY import Vector_field_XY
from diffractio.vector_fields_XYZ import Vector_field_XYZ
from diffractio.vector_fields_XZ import Vector_field_XZ
from diffractio.vector_fields_Z import Vector_field_Z

from diffractio.vector_masks_XY import Vector_mask_XY
from diffractio.vector_sources_XY import Vector_source_XY

from diffractio.utils_typing import NDArrayFloat

class Diffractio():
    """Class for unidimensional scalar fields.

    Args:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): Linear array with equidistant positions.
            The number of data is preferibly :math:`2^n`.
        self.wavelength (float): Wavelength of the incident field.
        self.u (numpy.array): Complex field. The size is equal to self.x.
        self.quality (float): Quality of RS algorithm.
        self.info (str): Description of data.
        self.type (str): Class of the field.
        self.date (str): Date when performed.
    """

    def __init__(self, 
                 kind: Options_Diffractio_kind,
                 frame: Options_Diffractio_frame,
                 x: NDArrayFloat | None = None, 
                 y: NDArrayFloat | None = None, 
                 z: NDArrayFloat | None = None, 
                 wavelength: float  = 0,
                 n_background: float = 1., 
                 info: str = ""):
        
        if kind == 'scalar':
            if x is not None and y is None and z is None:
                if frame == 'source':
                    self.__class__ =  Scalar_source_X
                    self.__init__(x, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ =  Scalar_mask_X
                    self.__init__(x, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_X
                    self.__init__(x, wavelength, n_background, info)
            elif x is not None and y is not None and z is None:
                if frame == 'source':
                    self.__class__ = Scalar_source_XY
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ = Scalar_mask_XY
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XY
                    self.__init__(x, y, wavelength, n_background, info)
            elif x is not None and y is None and z is not None:
                if frame == 'mask':
                    self.__class__ = Scalar_mask_XZ
                    self.__init__(x, z, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XZ
                    self.__init__(x, z, wavelength, n_background, info)
            elif x is not None and y is not None and z is not None:
                if frame == 'mask':
                    self.__class__ = Scalar_mask_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
                elif frame == 'field':
                    self.__class__ = Scalar_field_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
            elif x is None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Scalar_field_Z
                    self.__init__(z, wavelength, n_background, info)
            else:
                raise ValueError('frame must be source, mask or field')
        elif kind == 'vector':
            if x is not None and y is None and z is None:
                if frame == 'field':
                    self.__class__ =  Vector_field_X    
                    self.__init__(x, wavelength, n_background, info)  
            elif x is not None and y is not None and z is None:
                if frame == 'field':
                    self.__class__ = Vector_field_XY  
                    self.__init__(x, y, wavelength, n_background, info)
                elif frame == 'mask':
                    self.__class__ = Vector_mask_XY
                    self.__init__(x, y, wavelength, n_background, info)  
                elif frame == 'source':
                    self.__class__ = Vector_source_XY
                    self.__init__(x, y, wavelength, n_background, info) 
            elif x is not None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_XZ
                    self.__init__(x, z, wavelength, n_background, info)
            elif x is not None and y is not None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_XYZ
                    self.__init__(x, y, z, wavelength, n_background, info)
            elif x is None and y is None and z is not None:
                if frame == 'field':
                    self.__class__ = Vector_field_Z
                    self.__init__(z, wavelength, n_background, info)
            else:
                raise ValueError('frame must be fields, source or mask')
        else:
            raise ValueError('kind must be scalar or vector')
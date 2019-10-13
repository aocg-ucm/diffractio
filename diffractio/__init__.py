# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level package for Python Scalar and vector diffraction and interference."""
"""
Diffractio: A scientific computing package for Optical Interference and Diffraction in Python
======================================================================================================

Contents
--------
diffractio presents the following subpackages:

scalar unidimensional fields
    * scalar_fields_X  (propagation and general functions)
    * scalar_sources_X  (light sources)
    * scalar_masks_X (scalar masks)

scalar unidimensional X fields propagated in Z direction
    * scalar_fields_XZ  (propagation and general functions)
    * scalar_masks_XZ (scalar masks)
    * sources are acquired using scalar_sources_X

scalar bidimensional XY fields
    * scalar_fields_XY  (propagation and general functions)
    * scalar_sources_XY  (light sources)
    * scalar_masks_XY (scalar masks)

scalar bidimensional XY fields propagated in Z direction
    * scalar_fields_XYZ  (propagation and general functions)
    * scalar_masks_XYZ (scalar masks)
    * sources are acquired using scalar_sources_XY
"""

import multiprocessing

import numpy as np
import scipy as sp
from matplotlib import rcParams, cm
import matplotlib.pyplot as plt

__author__ = """Luis Miguel Sanchez Brea"""
__email__ = 'optbrea@ucm.es'
__version__ = '0.0.5'

name = 'diffractio'

um = 1.
mm = 1000. * um
nm = um / 1000.
degrees = np.pi / 180.
s = 1.
seconds = 1.

eps = 1e-6
num_decimals = 4
num_max_processors = multiprocessing.cpu_count()

no_date = False  # for test folder

params_drawing = dict()
params_drawing['color_intensity'] = cm.hot  # cm.gist_heat  #cm.hot
params_drawing['color_amplitude'] = cm.magma
params_drawing['color_phase'] = cm.seismic  # cm.hsv
params_drawing['color_real'] = cm.seismic

print("number of processors: {}".format(num_max_processors))

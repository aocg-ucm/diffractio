# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        diffractio_init.py
# Purpose:     Initialization for diffractio package
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Copyright:   AOCG / UCM
# Licence:     GPL
# ----------------------------------------------------------------------


"""Top-level package for Python Scalar and vector diffraction and interference.

Diffractio: A scientific computing package for Scalar and Vector Optical Interference and Diffraction in Python
==================================================================================================================

__init__.py: 52 lines
utils_tests.py: 206 lines
vector_masks_XY.py: 622 lines
scalar_fields_X.py: 1663 lines
scalar_sources_X.py: 211 lines
utils_common.py: 224 lines
scalar_masks_X.py: 975 lines
scalar_masks_XYZ.py: 279 lines
utils_typing.py: 59 lines
scalar_masks_XY.py: 2593 lines
vector_fields_Z.py: 774 lines
vector_fields_XZ.py: 1834 lines
scalar_masks_XZ.py: 1582 lines
config.py: 50 lines
vector_fields_XY.py: 1923 lines
utils_drawing.py: 454 lines
utils_dxf.py: 187 lines
scalar_fields_XZ.py: 2054 lines
scalar_fields_Z.py: 460 lines
vector_fields_XYZ.py: 661 lines
utils_math.py: 982 lines
utils_optics.py: 1218 lines
vector_sources_XY.py: 410 lines
utils_drawing3D.py: 623 lines
scalar_fields_XYZ.py: 1510 lines
scalar_fields_XY.py: 2734 lines
utils_multiprocessing.py: 169 lines
scalar_sources_XY.py: 537 lines
vector_fields_X.py: 787 lines

Total number of lines in Python files: 26384  (2024/09/26)

"""

import datetime
import multiprocessing
from matplotlib import cm, rcParams
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from .config import CONF_DRAWING

__author__ = """Luis Miguel Sanchez Brea"""
__email__ = 'optbrea@ucm.es'
__version__ = '0.3.1'
name = 'diffractio'

um = 1.
mm = 1000. * um
nm = um / 1000.
degrees = np.pi / 180.
s = 1.
seconds = 1.

eps = 1e-6
num_decimals = 4

no_date = False  # for test

now = datetime.datetime.now()
date_test = now.strftime("%Y-%m-%d_%H")

num_max_processors = multiprocessing.cpu_count()

rcParams['figure.dpi'] = 75

# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        config.py
# Purpose:     Configuration settings for diffractio package
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Copyright:   AOCG / UCM
# Licence:     GPL
# ----------------------------------------------------------------------


"""
Configuration file. Standard diffractio units are um: um = 1.
"""

from typing import Literal
from matplotlib import cm
import numpy as np

# Configuration params for drawings
CONF_DRAWING = dict()
CONF_DRAWING['color_intensity'] = cm.gist_heat  # cm.gist_heat  #cm.hot
CONF_DRAWING['color_amplitude'] = cm.jet
CONF_DRAWING['color_amplitude_sign'] = cm.seismic
CONF_DRAWING['color_phase'] = cm.twilight  # twilight .twilight hsv
CONF_DRAWING['color_real'] = cm.seismic
CONF_DRAWING['color_stokes'] = cm.seismic
CONF_DRAWING['color_n'] = cm.Blues
CONF_DRAWING[
    'percentage_intensity'] = 0.05  # percentage of intensity not shown in phase

# Kind parameters for drawing
Draw_refractive_index_Options = Literal['all', 'real', 'imag']
Draw_X_Options = Literal[ 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft']
Draw_Z_Options = Literal['amplitude', 'intensity', 'field', 'phase']
Draw_XZ_Options = Literal['amplitude', 'intensity', 'phase', 'real']
Draw_XY_Options = Literal['amplitude', 'intensity', 'phase', ' field', 'real_field', 'contour']
Draw_XYZ_Options = Literal['intensity', 'refractive_index']
Draw_pyvista_Options = Literal['volume', 'clip', 'slices', 'projections']
Draw_interactive_Options = Literal['intensity', 'amplitude', 'phase']
Draw_Vector_X_Options = Literal[ 'intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'stokes']
Draw_Vector_XY_Options = Literal['intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'stokes', 'param_ellipse', 'ellipses']
Draw_Vector_XZ_Options = Literal[ 'intensity', 'intensities', 'intensities_rz', 'phases', 'fields', 'stokes']
number_types = (int, float, complex, np.int32, np.float64)
empty_types = ([], '', None)
Save_mask_Options = Literal['amplitude', 'phase', 'intensity']
Options_add = Literal['source', 'mask', 'phases', 'no_overlap', 'distances']

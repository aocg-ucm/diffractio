"""
Configuration file. Standard py_lab units are mm.
"""

import numpy as np
from matplotlib import cm

# Configuration params for drawings
CONF_DRAWING = dict()
CONF_DRAWING['color_intensity'] = cm.gist_heat  # cm.gist_heat  #cm.hot
CONF_DRAWING['color_amplitude'] = cm.jet
CONF_DRAWING['color_amplitude_sign'] = cm.seismic
CONF_DRAWING['color_phase'] = cm.twilight  # .twilight hsv
CONF_DRAWING['color_real'] = cm.seismic
CONF_DRAWING['color_stokes'] = cm.seismic
CONF_DRAWING['percentaje_intensity'] = 0.00  # percentaje of intensity not shown in phase

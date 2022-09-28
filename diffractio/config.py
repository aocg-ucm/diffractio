# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file. Standard py_lab units are mm.
"""

from matplotlib import cm

# Configuration params for drawings
CONF_DRAWING = dict()
CONF_DRAWING['color_intensity'] = cm.gist_heat  # cm.gist_heat  #cm.hot
CONF_DRAWING['color_amplitude'] = cm.jet
CONF_DRAWING['color_amplitude_sign'] = cm.seismic
CONF_DRAWING['color_phase'] = cm.twilight  # twilight .twilight hsv
CONF_DRAWING['color_real'] = cm.seismic
CONF_DRAWING['color_stokes'] = cm.seismic
CONF_DRAWING['percentage_intensity'] = 0.005 # percentage of intensity not shown in phase

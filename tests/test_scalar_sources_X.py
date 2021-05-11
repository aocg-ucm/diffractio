# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:        scalar_sources_X.py
# Purpose:     1D light sources tests
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2019/01/30
#
# Licence:     GPL
# -------------------------------------------------------------------------------

import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, um
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.utils_tests import comparison, save_figure_test
from numpy import linspace

# from functools import wraps


if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_sources_X"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_sources_X(object):
    # def saving_data(f):
    #     @wraps(f)
    #     def wrapped(inst, *args, **kwargs):
    #         func_name = sys._getframe().f_code.co_name
    #         filename = '{}{}'.format(newpath, func_name)
    #         print(filename)
    #
    #         u0 = f(inst, *args, **kwargs)
    #         u0.save_data(filename=filename+'.npz')
    #         save_figure_test(newpath, func_name)
    #         assert True
    #
    #     assert True
    #     #return wrapped

    def test_plane_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-1000 * um, 1000 * um, 512)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_wave(theta=1 * degrees, z0=0 * um)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gauss_beam(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 2048)
        wavelength = .5 * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(
            A=1, x0=0 * um, z0=-2000 * um, w0=25 * um, theta=0 * degrees)

        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_spherical_wave_convergent(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 1024)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.spherical_wave(
            A=1, x0=0 * um, z0=5 * mm, radius=200 * um, mask=True)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_spherical_wave_divergent(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 1024)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.spherical_wave(
            A=1, x0=0 * um, z0=-5 * mm, radius=200 * um, mask=True)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_plane_waves_several_inclined(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 1024)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.plane_waves_several_inclined(
            A=1, num_beams=5, max_angle=5 * degrees)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_plane_waves_dict(self):
        assert True

    def test_gauss_beams_several_parallel(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 1024)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beams_several_parallel(
            A=1,
            num_beams=5,
            w0=50 * um,
            z0=0 * um,
            x_central=0 * um,
            x_range=750 * um,
            theta=0 * degrees)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gauss_beams_several_inclined(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-500 * um, 500 * um, 1024)
        wavelength = 0.6328 * um

        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beams_several_inclined(
            A=1,
            num_beams=5,
            w0=250 * um,
            x0=0 * um,
            z0=0 * um,
            max_angle=5 * degrees)
        u0.draw(kind='field')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_interferences(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 2 * mm
        x0 = linspace(-length / 2, length / 2, 1024)
        wavelength0 = 0.6238 * um

        u1 = Scalar_source_X(x=x0, wavelength=wavelength0)
        u2 = Scalar_source_X(x=x0, wavelength=wavelength0)

        u1.gauss_beam(
            A=1, x0=0 * um, z0=0 * um, w0=250 * um, theta=.25 * degrees)
        u2.gauss_beam(
            A=1, x0=0 * um, z0=0 * um, w0=250 * um, theta=-.25 * degrees)

        u0 = u1 + u2
        u0.draw(kind='intensity')

        u0.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

from diffractio import degrees, no_date, np, um
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_tests import save_figure_test
from diffractio.vector_paraxial_sources_XY import Vector_paraxial_source_XY

path_base = "tests_results"
path_class = "vector_paraxial_sources_XY"

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)

polarization_x = [1, 0]
polarization_y = [0, 1]
polarization_right = [1, 1.j] / np.sqrt(2)
polarization_left = [1, -1.j] / np.sqrt(2)
polarization_45 = [1, 1] / np.sqrt(2)
polarization_m45 = [1, -1] / np.sqrt(2)


class Test_vector_sources_XY(object):

    def test_constant_wave(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 25 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 1 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.constant_wave(u=1, v=(1, 1j))

        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_interferences(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 100 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        u0 = Scalar_source_XY(x0, y0, wavelength)
        u0.plane_wave(A=1, theta=-1 * degrees, phi=0 * degrees)

        u1 = Scalar_source_XY(x0, y0, wavelength)
        u1.plane_wave(A=1, theta=1 * degrees, phi=0 * degrees)

        EM1 = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM1.constant_wave(u=u0, v=[1, 0])

        EM2 = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM1.constant_wave(u=u1, v=[1, 0])

        EM = EM1 + EM2

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_azimuthal_wave(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.azimuthal_wave(
            u=1, r0=(0 * um, 0 * um), radius=(length / 2, length / 2))

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_radial_wave(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.radial_wave(
            u=1, r0=(0 * um, 0 * um), radius=(length / 2, length / 2))

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 75 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um
        u = Scalar_source_XY(x0, y0, wavelength)
        u.gauss_beam(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(15 * um, 15 * um),
            theta=0. * degrees,
            phi=0 * degrees)
        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.constant_wave(u, v=(1, 1))

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss_radial(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 75 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um
        u = Scalar_source_XY(x0, y0, wavelength)
        u.gauss_beam(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(15 * um, 15 * um),
            theta=0. * degrees,
            phi=0 * degrees)
        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.azimuthal_wave(u)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_wave(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_wave(
            u=1, r0=(0 * um, 0 * um), m=1.5, fi0=0 * np.pi)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_wave_radial(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_wave_radial(
            u=1, r0=(0 * um, 0 * um), m=0.5, fi0=np.pi / 2)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_wave_hybrid(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_wave_hybrid(
            u=1, r0=(0 * um, 0 * um), m=1, n=3, fi0=np.pi / 2)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

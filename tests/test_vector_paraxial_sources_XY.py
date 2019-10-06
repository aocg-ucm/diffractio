# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

from diffractio import degrees, mm, np, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_tests import save_figure_test
from diffractio.vector_paraxial_masks_XY import Vector_paraxial_mask_XY
from diffractio.vector_paraxial_sources_XY import Vector_paraxial_source_XY

path_base = "tests_results"
path_class = "vector_paraxial_sources_XY"
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d_%H_%M_%S")
date = '0'

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
    def test_wave_from_scalar(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 25 * um
        num_data = 1024
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 1 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(6 * um, 6 * um),
            z0=0 * um,
            theta=0. * degrees,
            phi=0 * degrees)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.unique_mask(u1, v=polarization_right)

        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_wave_plane_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        # con esto definimos el field E
        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.plane_wave(
            A=1, v=polarization_45, theta=1 * degrees, phi=45 * degrees)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_interferences(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 100 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        EM1 = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM1.plane_wave(A=1, v=[1, 0], theta=-1 * degrees, phi=0 * degrees)

        EM2 = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM2.plane_wave(A=1, v=[1, 0], theta=1 * degrees, phi=0 * degrees)

        EM = EM1 + EM2

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_spherical_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 50 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.spherical_wave(A=1, z0=-5 * mm, radius=length / 2, mask=False)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.unique_mask(u1, v=polarization_x)
        EM.normalize()

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_radial_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.radial_wave(A=1, x_center=(0 * um, 0 * um), radius=length)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_transversal_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.transversal_wave(A=1, x_center=(0 * um, 0 * um), radius=length)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_transversal_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.transversal_wave(A=1, x_center=(0 * um, 0 * um), radius=length)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss_linear(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 75 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.gauss(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(15 * um, 15 * um),
            theta=0. * degrees,
            phi=0 * degrees,
            kind='polarization',
            v=[1, 0])

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss_circular(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 75 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.gauss(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(15 * um, 15 * um),
            theta=0. * degrees,
            phi=0 * degrees,
            kind='polarization',
            v=[1, 1j])

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss_radial(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 200 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.gauss(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(50 * um, 50 * um),
            theta=0. * degrees,
            phi=0 * degrees,
            kind='radial')

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_gauss_transversal(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 200 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 10

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.gauss(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(50 * um, 50 * um),
            theta=0. * degrees,
            phi=0 * degrees,
            kind='transversal')

        EM.draw(kind='fields')

        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_hermite_gauss_radial(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 100 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)

        EM.hermite_gauss_wave(
            A=1,
            r0=(0 * um, 0 * um),
            w=125 * um,
            m=[2],
            n=[2],
            c_mn=[1],
            kind='radial',
            v=[1, 0])

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_hermite_gauss_transversal(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 5

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        # EM.hermite_gauss_wave(A=1, w=75*um, m = [0,1], n = [0,1], c_mn = [1,1], kind='transversal')
        EM.hermite_gauss_wave(
            A=1, w=75 * um, m=[1], n=[1], c_mn=[1], kind='transversal')

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_hermite_gauss_transversal(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 5

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        # EM.hermite_gauss_wave(A=1, w=75*um, m = [0,1], n = [0,1], c_mn = [1,1], kind='transversal')
        EM.hermite_gauss_wave(
            A=1, w=75 * um, m=[1], n=[1], c_mn=[1], kind='transversal')

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_beam(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_beam(
            A=1, r0=(0 * um, 0 * um), m=1.5, fi0=0 * np.pi)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_beam_radial(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_beam_radial(
            A=1, r0=(0 * um, 0 * um), n=0.5, fi0=np.pi / 2)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

    def test_local_polarized_vector_beam_hybrid(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_beam_hybrid(
            A=1, r0=(0 * um, 0 * um), m=1, n=3, fi0=np.pi / 2)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')

        assert True

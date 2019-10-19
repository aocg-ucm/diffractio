# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_tests import save_figure_test
from diffractio.vector_paraxial_masks_XY import Vector_paraxial_mask_XY
from diffractio.vector_paraxial_sources_XY import Vector_paraxial_source_XY
from py_pol.jones_matrix import Jones_matrix

path_base = "tests_results"
path_class = "vector_paraxial_masks_XY"

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


class Test_vector_masks_XY(object):
    def test_unique_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        # mask escalar
        mask = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        mask.ring(
            r0=(0 * um, 0 * um),
            radius1=(25 * um, 25 * um),
            radius2=(75 * um, 75 * um),
            angle=0 * degrees)

        # mask vectorial
        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.unique_mask(mask=mask, v=polarization_left)
        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')
        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_equal_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        # mask escalar
        mask = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        mask.ring(
            r0=(0 * um, 0 * um),
            radius1=(25 * um, 25 * um),
            radius2=(75 * um, 75 * um),
            angle=0 * degrees)

        # mask vectorial
        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.equal_mask(mask=mask)
        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_complementary_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        mask = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        mask.fresnel_lens(
            r0=(0 * um, 0 * um),
            radius=(125 * um, 125 * um),
            focal=(2 * mm, 2 * mm),
            angle=0 * degrees,
            mask=True,
            kind='amplitude')

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.complementary_masks(mask=mask, v1=(1, 0), v2=(0, 1))
        EM.draw(kind='intensities')

        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_apply_polarization(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        PL = Jones_matrix('m0')
        PL.from_elements(0.9, 0, 0, 0.2 * np.exp(1j))

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.apply_polarization(PL)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_polarizer_linear(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.polarizer_linear(angle=0 * degrees)
        EM.draw(kind='fields')

        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_quarter_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.quarter_waveplate(angle=0 * degrees)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_half_wave(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.half_waveplate(angle=0 * degrees)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

    def test_polarizer_retarder(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        wavelength = 0.6328 * um

        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.polarizer_retarder(
            delay=90 * degrees, p1=0.9, p2=0.1, angle=0 * degrees)

        EM.draw(kind='fields')
        save_figure_test(newpath, func_name, add_name='_fields')

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_stokes')
        assert True

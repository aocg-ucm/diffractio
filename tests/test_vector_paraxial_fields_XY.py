# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime
import os
import sys

from diffractio import (degrees, epsilon_0, mm, mu_0, nm, np, plt, sp,
                        speed_of_light, um)
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_tests import save_figure_test
from diffractio.vector_paraxial_masks_XY import Vector_paraxial_mask_XY
from diffractio.vector_paraxial_sources_XY import Vector_paraxial_source_XY

path_base = "tests_results"
path_class = "Vector_paraxial_fields_XY"
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d_%H_%M_%S")
date = '0'

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)

v_lineal = (1, 0, 0)
v_circular = (1, 1j, 0) / np.sqrt(2)


class Test_Vector_paraxial_fields_XY(object):
    def test_save_load(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_clear_field(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_mul(self):
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
            mask=True,
            kind='amplitude')

        vc = Vector_paraxial_mask_XY(x0, y0, wavelength)
        vc.equal_masks(mask=mask)

        vp = Vector_paraxial_mask_XY(x0, y0, wavelength)
        vp.polarizer_linear(angle=0 * degrees)

        EM = vp
        EM.draw(kind='Scalar_field_XY')
        save_figure_test(newpath, func_name, add_name='_vc')

        EM2 = vc * vp
        EM2.draw(kind='Scalar_field_XY')
        save_figure_test(newpath, func_name, add_name='_vp')
        assert True

    def test_generar_field_vectorial1(self):
        """
        aquí vemos cómo se puede generar el field vectorial a partir
        de vueltas sucesivas a los fields E y H
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 512
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.radial_wave(A=1000, x_center=(0 * um, 0 * um), radius=length / 5)

        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_0')

        EM.draw(field='S', kind='field')
        save_figure_test(newpath, func_name, add_name='_1')
        EM.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2')

        assert True

    def test_generar_field_vectorial2(self):
        """
        Lo mismo que antes, pero con la operación utils, sin testeo
        """

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.gauss(
            A=1000,
            z=0 * um,
            w0=(25 * um, 25 * um),
            kind='polarization',
            v=[1, 0])
        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_0')
        EM.__vectorize__(z0=-1 * mm, n=1)
        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_00')
        EM.draw(field='S', kind='field')
        save_figure_test(newpath, func_name, add_name='_1')
        EM.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2')
        assert True

    def test_get(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_apply_mask(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

    def test_plane_wave_generacion(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 16
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        # con esto definimos el field E
        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.plane_wave(A=1, v=[1, 0, 0], theta=0 * degrees, phi=0 * degrees)

        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_0')
        EM.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_1')

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um),
            radius=(
                length / 2,
                length / 2,
            ),
            angle=0 * degrees)

        EM.apply_mask(t1)

        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_2')
        EM.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_3')

        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)

        num_data = 256
        length = 150 * um
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(
            A=200,
            r0=(0 * um, 0 * um),
            w0=(25 * um, 25 * um),
            z0=0 * um,
            theta=0. * degrees,
            phi=0 * degrees)

        EM = Vector_paraxial_mask_XY(x0, y0, wavelength)
        EM.unique_mask(u1, v=v_lineal)
        EMdeltaz = EM.__vectorize__(
            z0=-.25 * mm, n=1, Vector_paraxial_normal=np.array([0, 0, 1]))

        EM.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_1EH')

        EM.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2S')

        EM.check_validity_EH(EMdeltaz)

        EMz = EM.RS(z=.5 * mm)
        EMz.draw(kind='intensityEH')
        save_figure_test(newpath, func_name, add_name='_3EH_p')

        EMz.draw(field='S', kind='intensities')
        save_figure_test(newpath, func_name, add_name='_4S_p')

        EMz.check_validity_EH('')
        EMz.draw(field='S', kind='intensity', numElipses=15, amplification=1)
        save_figure_test(newpath, func_name, add_name='_5I')
        return True

    def test_polarization_states(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_polarization_ellipse(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_draw(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}.npz'.format(newpath, func_name)
        assert True

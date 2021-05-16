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

path_base = "tests_results"
path_class = "Vector_paraxial_fields_XY"

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)

v_lineal = (1, 0, 0)
v_circular = (1, 1j, 0) / np.sqrt(2)


class Test_Vector_paraxial_fields_XY(object):
    def test_save_load(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_clear_field(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)
        assert True

    def test_mul(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

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
            kind='amplitude')

        vc = Vector_paraxial_mask_XY(x0, y0, wavelength)
        vc.apply_scalar_mask(u_mask=mask)

        vp = Vector_paraxial_mask_XY(x0, y0, wavelength)
        vp.polarizer_linear(azimuth=0 * degrees)

        EM = vp
        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_vc')

        EM2 = vc * vp
        EM2.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_vp')
        assert True

    def test_generar_field_vectorial1(self):
        """
        aquí vemos cómo se puede generar el field vectorial a partir
        de vueltas sucesivas a los fields E y H
        """
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 512
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.azimuthal_wave(
            u=1, r0=(0 * um, 0 * um), radius=(length / 5, length / 5))

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_0')

        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2')

        assert True

    def test_generar_field_vectorial2(self):
        """
        Lo mismo que antes, pero con la operación utils, sin testeo
        """

        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 250 * um
        num_data = 256
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2

        u0 = Scalar_source_XY(x0, y0, wavelength)
        u0.gauss_beam(A=1, z0=0 * um, r0=(0, 0), w0=(25 * um, 25 * um))

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.constant_wave(u0, v=[1, 1])

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_0')
        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_1')
        assert True

    def test_plane_wave_generacion(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        length = 500 * um
        num_data = 16
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328

        # con esto definimos el field E
        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.constant_wave(u=1, v=[1, 0])

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_0')
        EM.draw(kind='intensities')
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

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_2')
        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_3')

        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        num_data = 256
        length = 150 * um
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_wave_radial(u=1,
                                              r0=(0 * um, 0 * um),
                                              m=1,
                                              fi0=0,
                                              radius=0.)
        EM.mask_circle()

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_1EH')

        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2S')

        EMz = EM.RS(z=.5 * mm)
        EMz.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_3EH_p')

        EMz.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_4S_p')

        return True

    def test_VRS(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}.npz'.format(newpath, func_name)

        num_data = 256
        length = 150 * um
        x0 = np.linspace(-length / 2, length / 2, num_data)
        y0 = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 2 * um

        EM = Vector_paraxial_source_XY(x0, y0, wavelength)
        EM.local_polarized_vector_wave_radial(u=1,
                                              r0=(0 * um, 0 * um),
                                              m=1,
                                              fi0=0,
                                              radius=0.)
        EM.mask_circle()

        EM.draw(kind='stokes')
        save_figure_test(newpath, func_name, add_name='_1EH')

        EM.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_2S')

        EMz = EM.VRS(z=.5 * mm)
        EMz.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_3EH_p')

        EMz.draw(kind='intensities')
        save_figure_test(newpath, func_name, add_name='_4S_p')

        return True

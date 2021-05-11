# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for scalar_fields_XYZ
"""

import datetime
import os
import sys

from diffractio import degrees, eps, mm, no_date, np, um
from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_masks_XYZ import Scalar_mask_XYZ
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_tests import comparison, save_figure_test

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_fields_XYZ"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_fields_XYZ(object):
    def test_save_load(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 100 * um
        numdata = 16  # 256
        x0 = np.linspace(-length / 2, length / 2, numdata)
        y0 = np.linspace(-length / 2, length / 2, numdata)
        z0 = np.linspace(-length / 2, length / 2, numdata)
        wavelength = 0.5 * um

        t1 = Scalar_field_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)

        t1.u = np.ones_like(t1.u)
        t1.save_data(filename=filename+'.npz', add_name='')

        save_figure_test(newpath, func_name, add_name='_save')
        del t1

        t2 = Scalar_field_XYZ(x=None, y=None, z=None, wavelength=None)
        t2.load_data(
            filename=filename+'.npz', verbose=True)

        save_figure_test(newpath, func_name, add_name='_load')
        assert True

    def test_clear_field(self):
        # func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        length = 100 * um
        numdata = 32  # 256
        x0 = np.linspace(-length / 2, length / 2, numdata)
        y0 = np.linspace(-length / 2, length / 2, numdata)
        z0 = np.linspace(-length / 2, length / 2, numdata)
        wavelength = 0.5 * um

        u0 = Scalar_field_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)

        proposal = 0 * u0.u

        u0.clear_field()
        solution = u0.u

        assert comparison(proposal, solution, eps)

    def test_other(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 100 * um
        numdata = 32  # 256
        x0 = np.linspace(-length / 2, length / 2, numdata)
        y0 = np.linspace(-length / 2, length / 2, numdata)
        wavelength = 0.5 * um

        period = 10 * um
        z_talbot = 2 * period**2 / wavelength

        z0 = np.linspace(2 * z_talbot, 6 * z_talbot, 32)

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(A=1, r0=(0, 0), z0=0, w0=(150 * um, 150 * um))

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.ronchi_grating(period=10 * um, x0=0 * um, angle=0 * degrees)
        t3 = u1 * t1

        uxyz = Scalar_field_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
        uxyz.incident_field(t3)
        uxyz.RS()
        uxyz.draw_XYZ(logarithm=False, normalize='maximum')

        uxyz.info = """info:
            test_other(self):
            """
        filename = uxyz.save_data(filename=filename+'.npz')
        del uxyz

        # u2 = scalar_fields_XYZ(None, None, None)
        # u2.load_data(
        #     filename=filename+'.npz', verbose=True)
        # u2.draw_XYZ(logarithm=False, normalize='maximum')
        # u2.save_data(filename=filename+'.npz')
        # save_figure_test(newpath, func_name)
        assert True

    def test_show_index_refraccion(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 50 * um
        numdataX = 64
        numdataZ = 64

        longitud = 50 * um

        x0 = np.linspace(-length / 2, length / 2, numdataX)
        y0 = np.linspace(-length / 2, length / 2, numdataX)
        z0 = np.linspace(-longitud, longitud, numdataZ)
        wavelength = 0.55 * um

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um), radius=(20 * um, 20 * um), angle=0 * degrees)

        uxyz = Scalar_mask_XYZ(
            x=x0, y=y0, z=z0, wavelength=wavelength, n_background=1., info='')
        uxyz.incident_field(t1)
        uxyz.cylinder(
            r0=(0 * um, 0 * um, 0),
            radius=(20 * um, 20 * um),
            length=longitud,
            refraction_index=1.5,
            axis=(0, 0, 0),
            angle=0 * degrees)

        uxyz.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_RS(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x0 = np.linspace(-25 * um, 25 * um, 32)
        y0 = np.linspace(-25 * um, 25 * um, 32)
        z0 = np.linspace(100 * um, 500 * um, 32)
        wavelength = .6328 * um

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um), radius=(10 * um, 10 * um), angle=0 * degrees)

        uxyz = Scalar_mask_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
        uxyz.sphere(
            r0=(0 * um, 0 * um, 200 * um),
            radius=(50 * um, 50 * um, 50 * um),
            refraction_index=2 + 1j,
            angles=(0 * degrees, 0 * degrees, 0 * degrees))

        uxyz.incident_field(u0=t1)

        uxyz.RS(verbose=True, num_processors=4)
        uxyz.draw_XYZ(
            kind='intensity', logarithm=False, normalize='maximum')
        uxyz.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_propagacion_RS_focal_lens(self):
        func_name = sys._getframe().f_code.co_name
        # filename = '{}{}'.format(newpath, func_name)

        length = 500 * um
        x0 = np.linspace(-length / 2, length / 2, 16)
        y0 = np.linspace(-length / 2, length / 2, 16)
        z0 = np.linspace(2 * mm, 7 * mm, 16)
        wavelength = 0.6328 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(A=1, r0=(0, 0), z0=0, w0=(150 * um, 150 * um))

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.ronchi_grating(period=20 * um, x0=0 * um, angle=0 * degrees)
        t2 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t2.lens(
            r0=(0 * um, 0 * um),
            radius=(200 * um, 200 * um),
            focal=(5 * mm, 5 * mm),
            angle=0 * degrees)

        t3 = u1 * t1 * t2

        uxyz = Scalar_field_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
        uxyz.incident_field(t3)
        uxyz.RS()

        u_xy = uxyz.to_Scalar_field_XY(
            iz0=None, z0=3 * mm, is_class=True, matrix=False)
        u_xy.draw(kind='intensity')

        uxyz.draw_XY(
            z0=2 * mm, filename='{}a_{}'.format(newpath, func_name))
        uxyz.draw_XY(
            z0=4.5 * mm, filename='{}b_{}'.format(newpath, func_name))
        uxyz.draw_XZ(
            y0=0 * mm,
            logarithm=True,
            filename='{}_c{}'.format(newpath, func_name))
        # uxyz.draw_XYZ(logarithm=False, normalize='maximum')
        # uxyz.draw_volume(logarithm=1, normalize='maximum', maxintensity=None)

        return uxyz

    def test_propagacion_RS_Talbot_video(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 200 * um
        x0 = np.linspace(-length / 2, length / 2, 16)
        y0 = np.linspace(-length / 2, length / 2, 16)
        wavelength = 0.5 * um

        period = 10 * um
        z_talbot = 2 * period**2 / wavelength

        z0 = np.linspace(2 * z_talbot, 6 * z_talbot, 16)

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(A=1, r0=(0, 0), z0=0, w0=(150 * um, 150 * um))
        u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.ronchi_grating(period=10 * um, x0=0 * um, angle=0 * degrees)
        t3 = u1 * t1

        uxyz = Scalar_field_XYZ(x=x0, y=y0, z=z0, wavelength=wavelength)
        uxyz.incident_field(t3)
        uxyz.RS()

        uscalar_fields_XY = uxyz.to_Scalar_field_XY(z0=3 * mm)
        uscalar_fields_XY.draw(kind='intensity', cut_value=None)

        uxyz.draw_XY(
            z0=2 * mm, filename='{}a_{}'.format(newpath, func_name))
        uxyz.draw_XY(
            z0=4.5 * mm, filename='{}b_{}'.format(newpath, func_name))
        uxyz.draw_XZ(
            y0=0 * mm,
            logarithm=True,
            filename='{}_c{}'.format(newpath, func_name))
        # uxyz.draw_XYZ(logarithm=False, normalize='maximum')
        # uxyz.draw_volume(logarithm=1, normalize='maximum', maxintensity=None)
        uxyz.save_data(
            filename=filename+'.npz', add_name='')

        uxyz.video(filename=filename + '.avi', kind='intensity', frame=True)

        return uxyz

    def test_BPM(self):
        """
        cylinder torcido que hace de lente en una direccion solo
        """
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 80 * um
        numdataX = 32
        numdataZ = 32
        longitud = 400 * um
        radiusFibra = 10 * um
        x0 = np.linspace(-length / 2, length / 2, numdataX)
        y0 = np.linspace(-length / 2, length / 2, numdataX)
        z0 = np.linspace(0, longitud, numdataZ)
        wavelength = 0.55 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.gauss_beam(
            A=1,
            r0=(0 * um, 0 * um),
            w0=(radiusFibra / 1, radiusFibra / 1),
            theta=0. * degrees,
            phi=0 * degrees)
        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um),
            radius=(radiusFibra, radiusFibra),
            angle=0 * degrees)
        u2 = u1 * t1

        uxyz = Scalar_mask_XYZ(
            x=x0, y=y0, z=z0, wavelength=wavelength, n_background=1., info='')
        uxyz.incident_field(u2)
        # uxyz.cylinder(
        #     r0=(0 * um, 0 * um, 0),
        #     radius=(2 * radiusFibra, 2 * radiusFibra),
        #     length=longitud,
        #     refraction_index=2)

        uxyz.BPM()

        uxyz.draw_XYZ(logarithm=True, normalize='maximum')
        uxyz.draw_refraction_index()
        uxyz.draw_XZ(
            y0=0.01, logarithm=True, normalize='false', draw_borders=False)

        # uxyz.draw_volume(logarithm=True, normalize='maximum')

        uxyz.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True
        return uxyz

    def test_video(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 80 * um
        numdataX = 32
        numdataZ = 32
        longitud = 50 * um
        radiusFibra = 10 * um
        x0 = np.linspace(-length / 2, length / 2, numdataX)
        y0 = np.linspace(-length / 2, length / 2, numdataX)
        z0 = np.linspace(0, longitud, numdataZ)
        wavelength = 2 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave()

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um),
            radius=(4 * radiusFibra, 2 * radiusFibra),
            angle=45 * degrees)
        u2 = u1 * t1

        uxyz = Scalar_mask_XYZ(
            x=x0, y=y0, z=z0, wavelength=wavelength, n_background=1., info='')
        uxyz.incident_field(u2)

        uxyz.BPM()

        uxyz.video(filename=filename + '.avi', kind='intensity', frame=False)
        uxyz.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_xy_2_xyz(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 200 * um
        numdata = 32
        x0 = np.linspace(-length / 2, length / 2, numdata)
        y0 = np.linspace(-length / 2, length / 2, numdata)
        z0 = np.linspace(.1 * mm, .2 * mm, 64)
        wavelength = 0.6328 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave()
        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.square(
            r0=(0 * um, 0 * um), size=(20 * um, 20 * um), angle=0 * degrees)
        t2 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t2.ronchi_grating(period=10 * um, x0=20 * um, angle=0 * degrees)

        u2 = u1 * t1 * t2
        u2.draw()
        fields_XY = []
        for i in range(len(z0)):
            u3 = u2.RS(z=z0[i], new_field=True)
            fields_XY.append(u3)

        z0p = np.linspace(1 * mm, 7 * mm, numdata)
        uxyz = Scalar_field_XYZ(x=x0, y=y0, z=z0p, wavelength=wavelength)
        uxyz.incident_field(u2)
        uxyz.xy_2_xyz(fields_XY, z0)

        # uxyz.draw_volume()

        uxyz.video(filename=filename + '.avi', kind='intensity', frame=True)
        u3.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_cut_resample(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        length = 50 * um
        numdataX = 32
        numdataZ = 32
        longitud = 512 * um
        radiusFibra = 25 * um
        x0 = np.linspace(-length, length, numdataX)
        y0 = np.linspace(-length, length, numdataX)
        z0 = np.linspace(0, longitud, numdataZ)
        wavelength = 0.55 * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave()
        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um),
            radius=(radiusFibra, radiusFibra),
            angle=0 * degrees)
        u2 = u1 * t1

        uxyz = Scalar_mask_XYZ(
            x=x0, y=y0, z=z0, wavelength=wavelength, n_background=1., info='')
        uxyz.incident_field(u2)
        uxyz.sphere(
            r0=(0 * um, 0 * um, radiusFibra),
            radius=(radiusFibra, radiusFibra, radiusFibra),
            refraction_index=2,
            angles=(0, 0, 0))

        # uxyz.draw_refraction_index()
        uxyz.draw_XYZ()
        uxyz2 = uxyz.cut_resample(
            x_limits=(-25 * um, 25 * um),
            y_limits=(-25 * um, 25 * um),
            z_limits=(0 * um, 250 * um),
            num_points=[],
            new_field=True,
            interp_kind=(3, 1))

        uxyz2.draw_XYZ()
        # uxyz2.draw_refraction_index()
        uxyz2.save_data(filename=filename+'.npz')
        save_figure_test(newpath, func_name)
        assert True

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests for scalar_masks_X"""

import datetime
import os
import sys

from diffractio import degrees, mm, no_date, np, um
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.utils_tests import save_figure_test
from numpy import loadtxt

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "scalar_masks_X"

newpath = "{}_{}/{}/".format(path_base, date, path_class)

if not os.path.exists(newpath):
    os.makedirs(newpath)


class Test_Scalar_masks_X(object):
    def test_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 128
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.slit(x0=0, size=100 * um)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_double_slit(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 128
        length = 25 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.double_slit(x0=0, size=5 * um, separation=15 * um)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_two_levels(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 128
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.two_levels(level1=0, level2=.5, x_edge=0)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_gray_scale(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(0, 250 * um, 1024)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.gray_scale(num_levels=16, levelMin=2, levelMax=3)
        t1.draw(kind='amplitude')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_prism(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.prism(x0=0, n=1.5, anglex=1 * degrees)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_biprism_fresnel(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024
        length = 500 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.biprism_fresnel(angle=1 * degrees,
                           x0=100 * um,
                           radius=125,
                           mask=True)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_biprism_fresnel_nh(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.biprism_fresnel_nh(x0=0, width=100 * um, height=5 * um, n=1.5)

        t1.draw(kind='phase')
        t1.save_data(filename=filename + '.npz')
        pass

    def test_lens(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 128
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.lens(x0=0 * um, radius=100 * um, focal=5 * mm)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_fresnel_lens(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.fresnel_lens(x0=0 * um,
                        radius=100 * um,
                        focal=.25 * mm,
                        mask=True,
                        kind='amplitude',
                        phase=np.pi)
        t1.draw(kind='amplitude')
        save_figure_test(newpath, func_name, '_amplitude')

        t1.fresnel_lens(x0=0 * um,
                        radius=100 * um,
                        focal=.25 * mm,
                        mask=True,
                        kind='phase',
                        phase=np.pi)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name, '_phase')
        assert True

    def test_roughness(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 128
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)
        t1.roughness(t=15 * um, s=.2 * um)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_dust_different_sizes(self):
        # TODO: does not work properly

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        wavelength = 0.6328 * um
        x = np.linspace(-500 * um, 500 * um, 2048)

        t1 = Scalar_mask_X(x, wavelength)
        t1.dust_different_sizes(percentaje=0.2, size=20 * um, std=5 * um)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_dust(self):

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        wavelength = 0.6328 * um
        x = np.linspace(-500 * um, 500 * um, 2048 * 8)

        t1 = Scalar_mask_X(x, wavelength)
        t1.dust(percentaje=0.9, size=20 * um)
        t1.draw()

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_sine_grating(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um
        period = 40 * um
        red = Scalar_mask_X(x, wavelength)
        red.sine_grating(period=period, amp_min=0, amp_max=1, x0=0 * um)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_redRonchi(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um
        red = Scalar_mask_X(x, wavelength)
        red.ronchi_grating(period=50 * um, x0=0 * um, fill_factor=0.75)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_redBinaria_amplitude(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um
        red = Scalar_mask_X(x, wavelength)
        red.binary_grating(period=50 * um,
                           amin=0.25,
                           amax=0.5,
                           phase=np.pi,
                           x0=25 * um,
                           fill_factor=0.25)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_redBinaria_phase(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um
        period = 40 * um
        red = Scalar_mask_X(x, wavelength)
        red.binary_grating(period=period,
                           amin=1,
                           amax=1,
                           phase=np.pi / 2,
                           x0=0,
                           fill_factor=0.5)
        red.draw(kind='phase')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_redBlazed(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um
        period = 125 * um
        red = Scalar_mask_X(x, wavelength)
        red.blazed_grating(x0=0, period=period, height=2 * um, n=1.5)
        red.draw(kind='phase')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_mask_from_function(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024
        length = 250 * um
        x = np.linspace(-length / 2, length / 2, num_data)
        wavelength = 0.6328 * um

        f1 = "R1-h1+np.sqrt(R1**2-(self.x-x0)**2)"
        f2 = "R2-h2+np.sqrt(R2**2-(self.x-x0)**2)"
        v_globals = {
            'R1': 5 * mm,
            'R2': 1 * mm,
            'x0': 0 * um,
            'y0': 0 * um,
            'h1': 1 * mm,
            'h2': -1 * mm
        }
        index = 1.5
        print(v_globals)

        t1 = Scalar_mask_X(x, wavelength)
        t1.mask_from_function(x0=0 * um,
                              index=index,
                              f1=f1,
                              f2=f2,
                              v_globals=v_globals,
                              radius=100 * um,
                              mask=True)
        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_mask_from_array(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 1024 * 8
        x = np.linspace(-1 * mm, 1 * mm, num_data)
        wavelength = 0.6328 * um

        t1 = Scalar_mask_X(x, wavelength)

        script_dir = os.path.dirname(__file__)
        rel_path1 = "profile1.txt"
        abs_file_path1 = os.path.join(script_dir, rel_path1)

        script_dir = os.path.dirname(__file__)
        rel_path2 = "profile2.txt"
        abs_file_path2 = os.path.join(script_dir, rel_path2)

        profile1 = loadtxt(abs_file_path1)
        profile2 = loadtxt(abs_file_path2)
        t1.mask_from_array(x0=0,
                           index=1.25,
                           array1=profile1 * mm,
                           array2=profile2 * mm,
                           interp_kind='quadratic',
                           radius=1.5 * mm,
                           mask=True)

        t1.draw(kind='phase')

        t1.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_chirped_grating_p(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(0 * um, 100 * um, 4096 * 4)
        wavelength = 0.6238 * um
        red = Scalar_mask_X(x, wavelength)
        conds = {
            'kind': 'amplitude_binary',
            'p0': 20 * um,
            'p1': 4 * um,
            'amp_min': 0,
            'amp_max': 1,
            'delta_x': 0,
            'phase_max': np.pi,
            'length': 0,  # equal to length of x
        }
        red.chirped_grating_p(**conds)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_chirped_grating_q(self):
        """chirped gratings with new definition
        'amplitude', 'phase', 'amplitude_binary', 'phase_binary'
        """

        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-100 * um, 100 * um, 4096 * 4)
        wavelength = 0.6238 * um
        red = Scalar_mask_X(x, wavelength)
        conds = {
            'kind': 'amplitude_binary',
            'p0': 20 * um,
            'p1': 4 * um,
            'amp_min': 0,
            'amp_max': 1,
            'delta_x': 0,
            'phase_max': np.pi,
            'length': 0,  # equal to length of x
        }
        red.chirped_grating_q(**conds)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_chirped_grating(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        x = np.linspace(-250 * um, 250 * um, 512)
        wavelength = 0.6238 * um

        fx = '10+20*(self.x/length)**2'

        red = Scalar_mask_X(x, wavelength)
        red.chirped_grating(kind='amplitude_binary',
                            p_x=fx,
                            length=500 * um,
                            x0=0 * um,
                            amp_max=1,
                            amp_min=0,
                            delta_x=0,
                            phase_max=np.pi)
        red.draw(kind='amplitude')

        red.save_data(filename=filename + '.npz')
        save_figure_test(newpath, func_name)
        assert True

    def test_code(self):
        func_name = sys._getframe().f_code.co_name
        filename = '{}{}'.format(newpath, func_name)

        num_data = 2048 * 8

        wavelength = 0.85 * um
        code = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
        anchura_bit = 80 * um
        x = np.linspace(0, anchura_bit * len(code), num_data)

        t1 = Scalar_mask_X(x, wavelength)
        t1.binary_code(kind='normal', code=code,
                       bit_width=anchura_bit, x0=0 * um)
        t1.draw()

        t1.save_data(filename=filename + '-normal' + '.npz')
        save_figure_test(newpath, func_name + '-normal')

        t2 = Scalar_mask_X(x, wavelength)
        t2.binary_code(x0=0 * um, kind='zeros', code=code, bit_width=anchura_bit)
        t2.draw()

        t2.save_data(filename=filename + '-zeros' + '.npz')
        save_figure_test(newpath, func_name + '-zeros')

        t3 = Scalar_mask_X(x, wavelength)
        t3.binary_code(x0=0, kind='ones', code=code, bit_width=anchura_bit)
        t3.draw()

        t3.save_data(filename=filename + '-ones' + '.npz')
        save_figure_test(newpath, func_name + '-ones')

        t4 = Scalar_mask_X(x, wavelength)
        t4.binary_code(x0=0, kind='abs_fag', code=code, bit_width=anchura_bit)
        t4.draw()

        t4.save_data(filename=filename + '-fag' + '.npz')
        save_figure_test(newpath, func_name)

        assert True

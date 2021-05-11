# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_field_X class and several functions for multiprocessing. It is required also for generating masks and fields.

The main atributes are:
    * self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
    * self.wavelength (float): wavelength of the incident field.
    * self.u (numpy.array): equal size than  x. complex field

There are also some secondary atributes:
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date


*Class for unidimensional scalar fields*

*Definition of a scalar field*
    * instantiation, clear_field, print
    * add, substract sources
    * multiply masks and sources
    * save and load data

*Functions for generation of masks*
    * insert masks, insert_array_masks - insert other masks inside the mask
    * filter -

*Propagation*
    * fft, ifft - fourier transform
    * RS - Rayleigh Sommerfeld. It allows amplification of the field
    * BPM - Beam propagation method

*Drawing functions*
    * draw

*Parameters:*
    * intensity, average intensity
    * get_edges_transitions (mainly for pylithography)

*Functions*
    * kernelRS, kernelRSinverse

*Multiprocessing*
    * polychromatic_multiprocessing
    * extended_source_multiprocessing
    * extended_polychromatic_source

"""

import copyreg
import time
import types

from numpy import (angle, array, concatenate, exp, linspace, pi, shape, sqrt,
                   zeros)
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft
from scipy.interpolate import interp1d
from scipy.special import hankel1

from . import degrees, mm, np, num_max_processors, plt
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw
from .utils_math import fft_filter, get_edges, nearest
from .utils_multiprocessing import (_pickle_method, _unpickle_method,
                                    execute_multiprocessing)
from .utils_optics import field_parameters, normalize

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Scalar_field_X(object):
    """Class for unidimensional scalar fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        wavelength (float): wavelength of the incident field
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): Linear array with equidistant positions.
            The number of data is preferibly :math:`2^n`.
        self.wavelength (float): Wavelength of the incident field.
        self.u (numpy.array): Complex field. The size is equal to self.x.
        self.quality (float): Quality of RS algorithm.
        self.info (str): Description of data.
        self.type (str): Class of the field.
        self.date (str): Date when performed.
    """

    def __init__(self, x=None, wavelength=None, n_background=1, info=""):
        self.x = x
        self.wavelength = wavelength
        self.n_background = n_background
        if x is not None:
            self.u = zeros(shape(self.x), dtype=complex)
        else:
            self.u = None
        self.quality = 0
        self.info = info
        self.type = 'Scalar_field_X'
        self.date = get_date()

    def __str__(self):
        """Represents main data of the atributes."""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - x:  {},   u:  {}".format(self.type, self.x.shape,
                                               self.u.shape))
        print(" - xmin:       {:2.2f} um,  xmax:      {:2.2f} um".format(
            self.x[0], self.x[-1]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""

    def __add__(self, other, kind='standard'):
        """Adds two Scalar_field_x. For example two light sources or two masks.

        Parameters:
            other (Scalar_field_X): 2nd field to add
            kind (str): instruction how to add the fields:
                - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1.
                - 'standard': add fields u3=u1+u2 and does nothing.

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """

        u3 = Scalar_field_X(self.x, self.wavelength)

        if kind == 'standard':
            u3.u = self.u + other.u

        elif kind == 'maximum1':
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            f1 = angle(self.u)
            f2 = angle(other.u)
            t3 = t1 + t2
            t3[t3 > 0] = 1.
            u3.u = t3 * exp(1j * (f1 + f2))

        return u3

    def __sub__(self, other):
        """Substract two Scalar_field_x. For example two light sources or two masks.

        Parameters:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO:
            It can be improved for maks (not having less than 1)
        """

        u3 = Scalar_field_X(self.x, self.wavelength)
        u3.u = self.u - other.u
        return u3

    def __mul__(self, other):
        """Multiply two fields. For example  :math: `u_1(x)= u_0(x)*t(x)`

        Parameters:
            other (Scalar_field_X): field to multiply

        Returns:
            Scalar_field_X: :math: `u_1(x)= u_0(x)*t(x)`
        """

        new_field = Scalar_field_X(self.x, self.wavelength)
        new_field.u = self.u * other.u
        return new_field

    def clear_field(self):
        """Removes the field so that self.u = 0. """
        self.u = np.zeros_like(self.u, dtype=complex)

    def save_data(self, filename, add_name='', description='', verbose=False):
        """Common save data function to be used in all the modules.
        The methods included are: npz, matlab


        Parameters:
            filename (str): filename
            add_name= (str): sufix to the name, if 'date' includes a date
            description (str): text to be stored in the dictionary to save.
            verbose (bool): If verbose prints filename.

        Returns:
            (str): filename. If False, file could not be saved.
        """
        try:
            final_filename = save_data_common(self, filename, add_name,
                                              description, verbose)
            return final_filename
        except:
            return False

    def load_data(self, filename, verbose=False):
        """Load data from a file to a Scalar_field_X.
            The methods included are: npz, matlab

        Parameters:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

        if verbose is True:
            print(dict0.keys())

    def cut_resample(self,
                     x_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind='quadratic'):
        """Cuts the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it does nothing.
        It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Parameters:
            x_limits (numpy.array): (x0,x1) - starting and final points to cut, if '' - takes the current limit x[0] and x[-1]
            num_points (int): it resamples x, and u [],'',0,None -> it leave the points as it is
            new_field (bool): if True it returns a new Scalar_field_X
            interp_kind (str): 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

        Returns:
            (Scalar_field_X): if new_field is True
        """

        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)

        if num_points not in ([], '', 0, None):
            x_new = linspace(x0, x1, num_points)
            f_interp_abs = interp1d(self.x,
                                    np.abs(self.u),
                                    kind=interp_kind,
                                    bounds_error=False,
                                    fill_value=0)

            f_interp_phase = interp1d(self.x,
                                      np.imag(self.u),
                                      kind=interp_kind,
                                      bounds_error=False,
                                      fill_value=0)

            u_new_abs = f_interp_abs(x_new)
            u_new_phase = f_interp_phase(x_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            x_new = self.x[i_s]
            u_new = self.u[i_s]

        if new_field is False:
            self.x = x_new
            self.u = u_new
        elif new_field is True:
            field = Scalar_field_X(x=x_new, wavelength=self.wavelength)
            field.u = u_new
            return field

    def incident_field(self, u0):
        """Incident field for the experiment. It takes a Scalar_source_X field.

        Parameters:
            u0 (Scalar_source_X): field produced by Scalar_source_X (or a X field)
        """
        self.u = u0.u

    def filter(self, size=0):
        """
        """

        from .scalar_masks_X import Scalar_mask_X  # Do not write up

        slit = Scalar_mask_X(self.x, self.wavelength)
        slit.slit(x0=0, size=size)
        self.u = fft_filter(self.u, slit.u)

    def insert_mask(self, t1, x0_mask1, clean=True, kind_position='left'):
        """Insert mask t1 in mask self. It is performed using interpolation.

        Parameters:
            t1 (Scalar_field_X): field X (shorter)
            x0_mask1 (float): location of starting point at t0 of init point of t1.
            clean (bool): if True remove previous fields, else substitute
            kind_position (str):  'left' 'center'
        """

        # reallocating the mask
        if kind_position == 'left':
            t1.x = t1.x - t1.x[0] + x0_mask1
        elif kind_position == 'center':
            t1.x = t1.x - (t1.x[0] + t1.x[-1]) / 2 + x0_mask1

        # interpolation is different for real and imag
        f_interp_real = interp1d(t1.x,
                                 np.real(t1.u),
                                 kind='nearest',
                                 bounds_error=False,
                                 fill_value=0)

        f_interp_imag = interp1d(t1.x,
                                 np.imag(t1.u),
                                 kind='nearest',
                                 bounds_error=False,
                                 fill_value=0)

        # interpolates all the range
        u_new_real = f_interp_real(self.x)
        u_new_imag = f_interp_imag(self.x)
        u_new = u_new_real + 1j * u_new_imag

        if clean is True:
            self.clear_field()
            self.u = u_new
        else:
            i_pos = (self.x > t1.x[0]) * (self.x < t1.x[-1])
            self.u[i_pos] = u_new[i_pos]

    def insert_array_masks(self, t1, x_pos, clean=True, kind_position='left'):
        """Insert several identical masks t1 in self.u according to positions x_pos

        Parameters:
            t1 (Scalar_field_X): mask to insert.
            x_pos (numpy.array): array with locations.
            clean (bool): elliminate preview fields in self.
            kind_position (str): 'left', 'center': positions are at left or center.
        """

        self.insert_mask(t1,
                         x_pos[0],
                         clean=clean,
                         kind_position=kind_position)

        for xi in x_pos[1:]:
            self.insert_mask(t1, xi, clean=False, kind_position=kind_position)

    def fft(self,
            z=None,
            shift=True,
            remove0=False,
            matrix=False,
            new_field=False,
            verbose=False):
        """Far field diffraction pattern using Fast Fourier Transform (FFT).

        Parameters:
            z (float): distance to the observation plane or focal of lens. If None the exit is in radians
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned.  If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            verbose (bool): if True, prints info

        Returns:
            (array or Scalar_field_X or None): FFT of the input field
        """

        ttf1 = fft(self.u)
        if remove0 is True:
            ttf1[0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Infor
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        if z in (None, '', [], 0):
            x_new = kx  # exit in angles (degrees)
            if verbose is True:
                print("z={}".format(z))
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        else:
            x_new = kx * z  # exit distances at a obsrvation plane z
            if verbose is True:
                print("z={}".format(z))
                print("x0={},x1={}".format(x_new[0], x_new[-1]))

        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = ttf1
            field_output.x = x_new
            return field_output
        else:
            self.u = ttf1
            self.x = x_new

    def ifft(self,
             z=None,
             shift=True,
             remove0=True,
             matrix=False,
             new_field=False,
             verbose=False):
        """Inverse Fast Fourier Transform (ifft) of the field.

        Parameters:
            z (float): distance to the observation plane or focal of lens
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            verbose (bool): if True, prints info

        Returns:
            (array or Scalar_field_X or None): FFT of the input field
        """
        ttf1 = ifft(self.u)
        if remove0 is True:
            ttf1[0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Infor
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * self.wavelength

        if z in (None, '', [], 0):
            x_new = kx  # exit in angles (radians)
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        else:
            x_new = kx * z  # exit distances at a obsrvation plane z
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = ttf1
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))
            field_output.x = x_new
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            if verbose is True:
                print("x0={},x1={}".format(x_new[0], x_new[-1]))

    def _RS_(self,
             z,
             n,
             matrix=False,
             new_field=True,
             fast=False,
             kind='z',
             xout=None,
             verbose=True):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask:

        :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)`

        The near field approach is performed according to  :math:`\mathbf{E}(x,y,z) = \frac{1}{i\lambda z}e^{i k z}\iint\mathbf{E}_{0}(\zeta,\eta)e^{i\frac{k}{2z}\left[\left(x-\zeta\right)^{2}+\left(y-\eta\right)^{2}\right]}d\zeta d\eta`

        If we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        Parameters:
            z (float): distance to observation plane. I z<0 inverse propagation is executed
            n (float): refraction index
            matrix (bool): if True returns a matrix with result. It is much faster than new_field=True
            new_field (bool): if False the computation goes to self.u. If True a new instance is produced
            fast (bool): Instead of using Hankle function for RS kernel uses exponential
            xout (numpy.array): for amplification
            verbose (bool): if True it writes to shell

        Returns:
            (Scalar_field_X or None): if New_field is True Scalar_field_X, else None

        References:
                Applied Optics vol 45 num 6 pp. 1102-1110 (2006)
        """

        if xout is None:
            xout = self.x[0]

        xout = self.x + xout - self.x[0]

        nx = len(xout)
        dx = xout[1] - xout[0]

        # parametro de quality
        dr_real = sqrt(dx**2)
        rmax = sqrt((xout**2).max())
        dr_ideal = sqrt((self.wavelength / n)**2 + rmax**2 + 2 *
                        (self.wavelength / n) * sqrt(rmax**2 + z**2)) - rmax
        self.quality = dr_ideal / dr_real

        if verbose is True:
            if (self.quality.min() > 1):
                print('Good result: factor {:2.2f}'.format(self.quality))
            else:
                print('- Needs denser sampling: factor {:2.2f}'.format(
                    self.quality))

        # matrix W para integracion simpson
        a = [2, 4]
        num_rep = int(round((nx) / 2) - 1)

        b = array(a * num_rep)
        W = concatenate(((1, ), b, (2, 1))) / 3.

        if float(nx) / 2 == round(nx / 2):  # es par
            i_central = num_rep + 1
            W = concatenate((W[:i_central], W[i_central + 1:]))

        # field
        U = zeros(2 * nx - 1, dtype=complex)
        U[0:nx] = array(W * self.u)

        xext = self.x[0] - xout[::-1]  # da la vuelta
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xout[0]))
        if z > 0:
            H = kernelRS(xext, self.wavelength, z, n, kind=kind, fast=fast)
        else:
            H = kernelRSinverse(xext,
                                self.wavelength,
                                z,
                                n,
                                kind=kind,
                                fast=fast)

        # calculo de la transformada de Fourier
        S = ifft(fft(U) * fft(H)) * dx
        Usalida = S[nx - 1:]

        # los calculos se pueden dejar en la instancia o crear un new field
        if matrix is True:
            return Usalida

        if new_field is True:
            field_output = Scalar_field_X(self.x, self.wavelength)
            # field_output.u = Usalida / sqrt(z)
            field_output.u = Usalida
            field_output.quality = self.quality
            return field_output
        else:
            # self.u = Usalida / sqrt(z)
            self.u = Usalida

    def RS(self,
           z=10 * mm,
           n=1,
           matrix=False,
           new_field=True,
           fast=False,
           amplification=1,
           kind='z',
           verbose=True):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Parameters:
            z (float): Distance to observation plane. if z<0 inverse propagation is executed
            n (float): Refraction index
            matrix (bool): If True, the result of the function is a numpy.array
            new_field (bool): if False the computation goes to self.u, if True a new instance is produced
            fast (bool): Instead of using Hankle function for RS kernel uses expotential
            amplification (int): number of frames in x direction
            kind (str): 'z'. In some circunstamces the function is used for other integrals
            verbose (bool): if True it writes to shell

        Returns:
            If New_field is True:  Scalar_field_X,
            If matrix is True: numpy.array()
            Else: None


        Info:
            This approach a quality parameter: If self.quality>1, propagation is right.
        """

        width_x = self.x[-1] - self.x[0]
        num_pixels = len(self.x)

        positions_x = -amplification * width_x / 2 + array(
            list(range(amplification))) * width_x

        x0 = linspace(-amplification * width_x / 2,
                      amplification * width_x / 2, num_pixels * amplification)

        u_field = np.zeros_like(x0, dtype=complex)
        qualities = np.zeros((amplification))
        for i, xi in zip(list(range(len(positions_x))),
                         np.flipud(positions_x)):
            u3 = self._RS_(z=z,
                           n=n,
                           matrix=False,
                           new_field=True,
                           fast=fast,
                           kind=kind,
                           xout=xi,
                           verbose=verbose)
            xshape = slice(i * num_pixels, (i + 1) * num_pixels)
            u_field[xshape] = u3.u
            qualities[i] = u3.quality

        if matrix is True:
            return u_field

        if new_field is True:
            field_output = Scalar_field_X(x=x0, wavelength=self.wavelength)
            field_output.u = u_field
            field_output.quality = qualities.min()
            return field_output
        else:
            self.x = x0
            self.u = u_field
            self.quality = qualities.min()

    def normalize(self, kind='intensity', new_field=False):
        """Normalizes the field so that intensity.max()=1.

        Parameters:
            kind (str): 'intensity, 'amplitude', 'logarithm'... other.. Normalization technique
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
        Returns
            u (numpy.array): normalized optical field
        """
        u_new = normalize(self.u, kind)

        if new_field is False:
            self.u = u_new
        else:
            field_output = Scalar_field_X(self.x, self.wavelength)
            field_output.u = u_new
            return field_output

    def MTF(self, kind='mm', has_draw=True):
        """Computes the MTF of a field,.

        Parameters:
            kind (str): 'mm', 'degrees'
            has_draw (bool): If True draws the MTF

        Returns:
            (numpy.array) fx: frequencies in lines/mm
            (numpy.array) mtf_norm: normalizd MTF
        """

        tmp_field = self.u
        x = self.x
        self.u = np.abs(self.u)**2
        MTF_field = self.fft(new_field=True, shift=True)

        i_center = int(len(MTF_field.x) / 2)

        mtf_norm = np.abs(MTF_field.u) / np.abs(MTF_field.u[i_center])

        # Image plane spacing
        delta_x = x[1] - x[0]
        # Nyquist frequencies on x and y direction
        frec_nyquist = 0.5 / delta_x
        # Defining spatial frequencies, 1000 passes um to mm
        fx = 1000 * linspace(-frec_nyquist, frec_nyquist, len(x))

        if kind == 'mm':
            frec = fx
            text_x = "$f_x (cycles/mm)$"
        elif kind == 'degrees':
            print("not implemented yet")
            frec = fx
            text_x = "$f_x (cycles/deg - not yet)$"

        if has_draw is True:
            plt.figure()
            plt.plot(frec, mtf_norm, 'k')
            plt.xlabel(text_x, fontsize=18)
            plt.ylabel("MTF", fontsize=18)

        self.u = tmp_field

        return fx, mtf_norm

    def intensity(self):
        """Intensity.

        Returns:
            (numpy.array): Intensity
        """

        intensity = (np.abs(self.u)**2)
        return intensity

    def average_intensity(self, verbose=False):
        """Returns the average intensity as: (np.abs(self.u)**2).sum() / num_data

        Parameters:
            verbose (bool): If True it prints the value of the average intensity.

        Returns:
            (float): average intensity.
        """
        num_data = len(self.x)
        average_intensity = (np.abs(self.u)**2).sum() / (num_data)
        if verbose is True:
            print("average intensity={} W/m").format(average_intensity)

        return average_intensity

    def get_edges(self,
                  kind_transition='amplitude',
                  min_step=0,
                  verbose=False,
                  filename=''):
        """Determine locations of edges for a binary mask.

        Parameters:
            kind_transition:'amplitude' 'phase'
                if we see the amplitude or phase of the field
            min_step: minimum step for consider a transition

        Returns:
            type_transition: array with +1, -1 with rasing or falling edges
            pos_transition: positions x of transitions
            raising: positions of raising
            falling: positions of falling
        """

        pos_transitions, type_transitions, raising, falling = get_edges(
            self.x, self.u, kind_transition, min_step, verbose, filename)
        return pos_transitions, type_transitions, raising, falling

    def draw(self,
             kind='intensity',
             logarithm=False,
             normalize=False,
             cut_value=None,
             filename='',
             scale=''):
        """Draws X field. There are several data from the field that are extracted, depending of 'kind' parameter.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
        """

        if self.x is None:
            print('could not draw file: self.x=None')
            return

        amplitude, intensity, phase = field_parameters(self.u)

        plt.figure()

        if kind == 'intensity':
            y = intensity
        elif kind == 'phase':
            y = phase
        elif kind in ('amplitude', 'fft', 'fill', 'field'):
            y = amplitude

        if kind in ('intensity', 'amplitude', 'fft', 'fill', 'field'):
            y = normalize_draw(y, logarithm, normalize, cut_value)

        if kind == 'field':
            plt.subplot(211)
            plt.plot(self.x, y, 'k', lw=2)
            plt.xlabel('$x\,(\mu m)$')
            plt.ylabel('$A\,(arb.u.)$')
            plt.xlim(left=self.x[0], right=self.x[-1])
            plt.ylim(bottom=0)

            plt.subplot(212)
            plt.plot(self.x, phase, 'k', lw=2)
            plt.xlabel('$x\,(\mu m)$')
            plt.ylabel('$phase\,(radians)$')
            plt.xlim(left=self.x[0], right=self.x[-1])

        elif kind in ('amplitude', 'intensity', 'phase'):
            plt.plot(self.x, y, 'k', lw=2)
            plt.xlabel('$x\,(\mu m)$')
            plt.ylabel(kind)
            plt.xlim(left=self.x[0], right=self.x[-1])

        elif kind == 'fft':
            plt.plot(self.x / degrees, y, 'k', lw=2)
            plt.xlim(left=self.x[0] / degrees, right=self.x[-1] / degrees)
            plt.xlabel('$\phi\,(degrees)$')
            plt.ylabel(kind)

        elif kind == 'fill':
            # this is for binary maks, as gratings and I0s.
            plt.fill_between(self.x, 0, amplitude)
            plt.xlabel('$x\,(\mu m)$')
            plt.ylabel(kind)
            plt.xlim(left=self.x[0], right=self.x[-1])

        if scale != '':
            plt.axis(scale)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        if kind == 'intensity':
            plt.ylim(bottom=0)
        elif kind == 'phase':
            plt.ylim(-pi, pi)


def kernelRS(x, wavelength, z, n=1, kind='z', fast=False):
    """Kernel for RS propagation. It uses the hankel tansform.

    There is a 'fast' version based on :math:`hk_1 = \sqrt{2/(\pi \, k \, R)}  e^{i  (k \, R - 3  \pi / 4)}` which approximates the result.

    Parameters:
        x (numpy.array): positions x
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refraction index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation
        fast (bool): If True  The approximation is much faster. According to https://dlmf.nist.gov/10.2#E5

    Returns:
        (complex array): kernel

    References:
        - https://dlmf.nist.gov/10.2#E5
        - F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
    """
    k = 2 * pi * n / wavelength
    R = sqrt(x**2 + z**2)

    if fast is False:
        hk1 = hankel1(1, k * R)
    elif fast is True:
        hk1 = sqrt(2 / (pi * k * R)) * exp(1.j * (k * R - 3 * pi / 4))

    if kind == 'z':
        return (0.5j * k * z / R) * hk1
    elif kind == 'x':
        return (0.5j * k * x / R) * hk1
    elif kind == '0':
        return (0.5j * k) * hk1


def kernelRSinverse(x, wavelength, z, n=1, kind='z', fast=False):
    """Kernel for inverse RS propagation. See also kernelRS

    Parameters:
        x (numpy.array): positions x
        wavelength (float): wavelength of incident fields
        z (float): distance for propagation
        n (float): refraction index of background
        kind (str): 'z', 'x', '0': for simplifying vector propagation
        fast (bool): If True  The approximation is much faster. According to https://dlmf.nist.gov/10.2#E5

    Returns:
        complex array: kernel
    """
    k = 2 * pi * n / wavelength
    R = sqrt(x**2 + z**2)

    if fast is False:
        hk1 = hankel1(1, k * R)
    elif fast is True:
        hk1 = sqrt(2 / (pi * k * R)) * exp(1.j * (k * R - 3 * pi / 4))

    if kind == 'z':
        return (-0.5j * k * z / R) * hk1
    elif kind == 'x':
        return (-0.5j * k * x / R) * hk1
    elif kind == '0':
        return (-0.5j * k) * hk1


def PWD_kernel(u, n, k0, k_perp2, dz):
    """
    Step for scalar (TE) Plane wave decomposition (PWD) algorithm.

    Parameters:
        u (np.array): fields
        n (np.array): refraction index
        k0 (float): wavenumber
        k_perp2 (np.array): transversal k**2
        dz (float): increment in distances

    Returns:
        numpy.array(): Field at at distance dz from the incident field

    References:
        1. Schmidt, S. et al. "Wave-optical modeling beyond the thin-element-approximation". 'Opt. Express' 24, 30188 (2016).
    """

    Ek = fftshift(fft(u))
    H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2))

    return ifft(fftshift(H * Ek))


def WPM_schmidt_kernel(u, n, k0, k_perp2, dz):
    """
    Kernel for fast propagation of WPM method

    Parameters:
        u (np.array): fields
        n (np.array): refraction index
        k0 (float): wavenumber
        k_perp2 (np.array): transversal k**2
        dz (float): increment in distances

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    refraction_indexes = np.unique(n)

    u_final = np.zeros_like(u, dtype=complex)
    for m, n_m in enumerate(refraction_indexes):
        # print (m, n_m)
        u_temp = PWD_kernel(u, n_m, k0, k_perp2, dz)
        Imz = (n == n_m)
        u_final = u_final + Imz * u_temp

    return u_final


def polychromatic_multiprocessing(function_process,
                                  wavelengths,
                                  spectrum,
                                  num_processors=num_max_processors,
                                  verbose=False):
    """
    It performs an analysis of polychromatic light. It needs a function with only one input parameter: wavelength.
    It determines the intensity for each wavelength and the final results is the summation of the intensities.

    Parameters:
        function_process (function): function with accepts params as input parameters:
        wavelengths (array): wavelengths in the spectrum
        spectrum (array): weights for the spectrum. if None: uniform spectrum, else array with the same dimension as wavelengths
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell

    Returns:
        intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        u_s (Scalar_field_X): fields for each wavelength
        time_proc (float): time interval in the processing
    """

    if not isinstance(spectrum, np.ndarray):
        spectrum = np.ones_like(wavelengths)

    if type(wavelengths) in (list, np.ndarray):
        u_s, time_proc = execute_multiprocessing(function_process, wavelengths,
                                                 num_processors, verbose)
        print(len(u_s))
        intensity = np.zeros_like(u_s[0].u, dtype=float)
        for i in range(len(wavelengths)):
            intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        intensity = intensity / spectrum.sum()
    else:
        time1 = time.time()
        u_s = function_process(wavelengths)
        time2 = time.time()
        intensity = np.abs(u_s.u)**2
        time_proc = time2 - time1

    return intensity, u_s, time_proc


def extended_source_multiprocessing(function_process,
                                    x0s,
                                    num_processors=num_max_processors,
                                    verbose=False):
    """
    It performs an analysis of extendes source light. It needs a function with only an input parameter, that is x0s positions of sources. It determines the intensity for each wavelength and it is added.

    Parameters:
        function_process (function): function with accepts params as input Parameters:
        x0s (array): positions of sources
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell

    Returns:
        - intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        - u_s (Scalar_field_X): fields for each wavelength
        - time_proc (float): time interval in the processing
    """

    if type(x0s) in (list, np.ndarray):
        u_s, time_proc = execute_multiprocessing(function_process, x0s,
                                                 num_processors, verbose)
        intensity = np.zeros_like(u_s[0].u, dtype=float)
        for i in range(len(x0s)):
            intensity = intensity + np.abs(u_s[i].u)**2
        intensity = intensity / len(x0s)
    else:
        time1 = time.time()
        u_s = function_process(x0s)
        time2 = time.time()
        intensity = np.abs(u_s.u)**2
        time_proc = time2 - time1
        if verbose is True:
            print("num_proc: {}, time={}".format(1, time_proc))

    return intensity, u_s, time_proc


def extended_polychromatic_source(function_process,
                                  x0s,
                                  wavelengths,
                                  spectrum,
                                  num_processors=num_max_processors,
                                  verbose=False):
    """ It performs an analysis of extendes source light. It needs a function with only an input parameter, that is x0s positions of sources. It determines the intensity for each wavelength and it is added.

    Parameters:
        function_process (function): function with accepts params as input Parameters:
        x0s (array): positions of sources
        wavelengths (array): wavelengths in the spectrum
        spectrum (array): weights for the spectrum. If None: uniform spectrum, else array with the same dimension as wavelengths
        num_processors (int): number of processors for the computation
        verbose (bool): if True send information to shell

    Returns:
        - intensity (array, complex): intensity = intensity + spectrum[i] * np.abs(u_s[i].u)**2
        - u_s (Scalar_field_X): fields for each wavelength
        - time_proc (float): time interval in the processing
    """

    dict_Parameters = []
    for i, wavelength in enumerate(wavelengths):
        for j, x0 in enumerate(x0s):
            dict_Parameters.append(
                dict(x0=x0, wavelength=wavelength, ij=(i, j)))

    u_s, time_proc = execute_multiprocessing(function_process, dict_Parameters,
                                             num_processors, verbose)
    intensity = np.zeros_like(u_s[0].u, dtype=float)
    for k in range(len(u_s)):
        # print( len(u_s), dict_Parameters[k]['ij'])
        i_wavelength = dict_Parameters[k]['ij'][0]
        intensity = intensity + spectrum[i_wavelength] * np.abs(u_s[k].u)**2
    intensity = intensity / (spectrum.sum() * len(x0s))

    return intensity, u_s, time_proc

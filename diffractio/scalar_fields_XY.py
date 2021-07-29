# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_field_XY class.

It can be considered an extension of Scalar_field_X for visualizing XY fields

For the case of Rayleigh sommefeld it is not necessary to compute all z positions but the final.

Nevertheless, for BPM method, intermediate computations are required. In this class, intermediate results are stored.

X,Y fields are defined using ndgrid (not with meshgrid, it is different).

It is required also for generating masks and fields.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.wavelength - wavdelength of the incident field. The field is monochromatic
    * self.u (numpy.array): equal size to x * y. complex field
    * self.X (numpy.array): equal size to x * y. complex field
    * self.Y (numpy.array): equal size to x * y. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date of execution

The magnitude is related to microns: `micron = 1.`

*Class for XY scalar fields*

*Definition of a scalar field*
    * instatiation,
    * save, load data
    * cut_resample, binarize, discretize
    * get_phase, get_amplitude, remove_amplitude, remove_phase, amplitude2phase, phase2amplitude

*Propagation*
    * fft, ifft, RS, RS_simple, RS_amplificacion

*Drawing functions*
    * draw, draw_profile,
    * video, progresion

*Parameters*
    * profile

*Functions outside the class*
    * draw_several_fields
    * draw2D
    * several_propagations
    * kernelRS, kernelRSinverse, kernelFresnel
"""

import datetime
import sys
import time

import matplotlib.animation as animation
import scipy.ndimage
from matplotlib import rcParams
from numpy import (angle, array, concatenate, cos, exp, flipud, linspace,
                   matrix, meshgrid, pi, real, shape, sin, sqrt, zeros)
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import RectBivariateSpline

from . import degrees, mm, np, params_drawing, plt, seconds, um
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import (draw2D, normalize_draw, prepare_drawing,
                            reduce_matrix_size)
from .utils_math import get_edges, get_k, ndgrid, nearest, rotate_image
from .utils_optics import beam_width_2D, field_parameters

try:
    import screeninfo
except:
    print("screeninfo not imported.")

try:
    import cv2
except:
    print("cv2 not imported. Function send_image_screen cannot be used")

percentaje_intensity = params_drawing['percentaje_intensity']


class Scalar_field_XY(object):
    """Class for working with XY scalar fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        y (numpy.array): linear array wit equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x=None, y=None, wavelength=None, info=""):
        self.x = x
        self.y = y
        self.wavelength = wavelength  # la longitud de onda
        if x is not None and y is not None:
            self.X, self.Y = meshgrid(x, y)
            self.u = zeros(shape(self.X), dtype=complex)
        else:
            self.X = None
            self.Y = None
            self.u = None
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XY'
        self.date = get_date()
        self.quality = 0
        self.params_drawing = params_drawing

    def __str__(self):
        """Represents main data of the atributes"""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - x:  {},   y:  {},   u:  {}".format(
            self.type, self.x.shape, self.y.shape, self.u.shape))
        print(" - xmin:       {:2.2f} um,  xmax:      {:2.2f} um".format(
            self.x[0], self.x[-1]))
        print(" - ymin:       {:2.2f} um,  ymax:      {:2.2f} um".format(
            self.y[0], self.y[-1]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""

    def __add__(self, other):
        """Adds two Scalar_field_x. For example two light sources or two masks

        Parameters:
            other (Scalar_field_X): 2 field to add

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """
        u3 = Scalar_field_XY(self.x, self.y, self.wavelength)
        u3.u = self.u + other.u
        return u3

    def __sub__(self, other):
        """Substract two Scalar_field_XY. For example two light sources or two masks

        Parameters:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO:
            It can be improved for maks (not having less than 1)
        """
        u3 = Scalar_field_XY(self.x, self.y, self.wavelength)
        u3.u = self.u - other.u
        return u3

    def __mul__(self, other):
        """Multiply two fields. For example  :math:`u_1(x)= u_0(x)*t(x)`

        Parameters:
            other (Scalar_field_X): field to multiply

        Returns:
            Scalar_field_X: :math:`u_1(x)= u_0(x)*t(x)`
        """
        new_field = Scalar_field_XY(self.x, self.y, self.wavelength)
        new_field.u = self.u * other.u

        return new_field

    def __rotate__(self, angle, position=None):
        """Rotation of X,Y with respect to position

        Parameters:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            y0 = (self.y[-1] + self.y[0]) / 2
        else:
            x0, y0 = position

        Xrot = (self.X - x0) * cos(angle) + (self.Y - y0) * sin(angle)
        Yrot = -(self.X - x0) * sin(angle) + (self.Y - y0) * cos(angle)
        return Xrot, Yrot

    def add(self, other, kind='standard'):
        """adds two Scalar_field_x. For example two light sources or two masks

        Parameters:
            other (Scalar_field_X): 2 field to add
            kind (str): instruction how to add the fields: - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1. - 'standard': add fields u3=u1+u2 and does nothing.

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """
        if kind == 'standard':
            u3 = Scalar_field_XY(self.x, self.y, self.wavelength)
            u3.u = self.u + other.u
        elif kind == 'maximum1':
            u3 = Scalar_field_XY(self.x, self.y, self.wavelength)
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            f1 = angle(self.u)
            f2 = angle(other.u)
            t3 = t1 + t2
            t3[t3 > 0] = 1.
            u3.u = t3 * exp(1j * (f1 + f2))

        return u3

    def rotate(self, angle, position=None):
        """Rotation of X,Y with respect to position. If position is not given, rotation is with respect to the center of the image

        Parameters:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            y0 = (self.y[-1] + self.y[0]) / 2
        else:
            x0, y0 = position

        center_rotation = y0, x0

        u_real_rotate = rotate_image(self.x, self.y, np.real(self.u),
                                     -angle * 180 / pi, center_rotation)
        u_imag_rotate = rotate_image(self.x, self.y, np.imag(self.u),
                                     -angle * 180 / pi, center_rotation)
        u_rotate = u_real_rotate + 1j * u_imag_rotate
        self.u = u_rotate

    def clear_field(self):
        """Removes the field: self.u=0.
        """
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

    def save_mask(self,
                  filename="",
                  kind='amplitude',
                  binarize=False,
                  info=""):
        """Create a mask in a file, for example, ablation or litography engraver

        Parameters:
            filename (str): file name
            kind (str): save amplitude or phase
            binarize (bool): If True convert the mask in (0,1) levels
            info (str): info of the mask

        Returns:
            float: area (in um**2)
        """

        # creo nombre para txt
        name = filename.split(".")
        nombreTxt = name[0] + ".txt"

        # creo image
        plt.figure()
        filter = np.abs(self.u) > 0

        if kind == 'amplitude':
            mask = np.abs(self.u)

        elif kind == 'phase':
            mask = np.angle(self.u)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask * filter

        if binarize is True:
            mask_min = mask.min()
            mask_max = mask.max()
            mask_mean = (mask_max + mask_min) / 2

            mask2 = np.zeros_like(mask)
            mask2[mask < mask_mean] = 0
            mask2[mask >= mask_mean] = 1

            mask = mask2

        plt.imsave(filename, mask, cmap='gray', dpi=300, origin='lower')
        plt.close()

        # creo txt con data importantes
        ofile = open(nombreTxt, "w")
        ofile.write("nombre de archivo %s\n" % filename)
        ofile.write("fecha: {}\n".format(datetime.date.today()))
        if info is not None:
            ofile.write("\ninfo:\n")
            ofile.write(info)
        ofile.write("\n\n")
        ofile.write("length de la mask: %i x %i\n" %
                    (len(self.x), len(self.y)))
        ofile.write("x0 = %f *um, x1 = %f *um, Deltax = %f *um\n" %
                    (self.x.min(), self.x[-1], self.x[1] - self.x[0]))
        ofile.write("y0 = %f *um, y1 = %f *um, Deltay = %f *um\n" %
                    (self.y.min(), self.y[-1], self.y[1] - self.y[0]))
        ofile.write("\nlongitud de onda = %f *um" % self.wavelength)
        ofile.close()

        return mask

    def cut_resample(self,
                     x_limits='',
                     y_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind=(3, 1)):
        """it cut the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Parameters:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut. if '' - takes the current limit y[0] and y[-1]
            num_points (int): it resamples x, y and u. [],'',,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XY
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            # used only for resampling
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)
        # new_num_points = i_x1 - i_x0
        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            f_interp_abs = RectBivariateSpline(self.y,
                                               self.x,
                                               np.abs(self.u),
                                               kx=kxu,
                                               ky=kxu,
                                               s=0)
            f_interp_phase = RectBivariateSpline(self.y,
                                                 self.x,
                                                 np.angle(self.u),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            u_new_abs = f_interp_abs(x_new, y_new)
            u_new_phase = f_interp_phase(x_new, y_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            X_new, Y_new = np.meshgrid(x_new, y_new)
            u_new = self.u[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.u = u_new
            self.X = X_new
            self.Y = Y_new
        elif new_field is True:
            field = Scalar_field_XY(x=x_new,
                                    y=y_new,
                                    wavelength=self.wavelength)
            field.u = u_new
            return field

    def incident_field(self, u0):
        """Incident field for the experiment. It takes a Scalar_source_X field.

        Parameters:
            u0 (Scalar_source_X): field produced by Scalar_source_X (or a X field)
        """
        self.u = u0.u

    def fft(self,
            z=10 * mm,
            shift=True,
            remove0=True,
            matrix=False,
            new_field=False):
        """Fast Fourier Transform (FFT) of the field.
        TODO: change to the frequency domain.

        Parameters:
            z (float): distance to the observation plane or focal of lens
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        """

        ttf1 = fft2(self.u)

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)

        kx = linspace(-freq_nyquist_x, freq_nyquist_x, num_x) * z
        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = linspace(-freq_nyquist_y, freq_nyquist_y, num_y) * z

        if new_field is True:
            field_output = Scalar_field_XY(self.x, self.y, self.wavelength)
            field_output.x = kx
            field_output.y = ky

            field_output.X, field_output.Y = ndgrid(field_output.x,
                                                    field_output.y)
            field_output.u = ttf1

            return field_output
        else:
            self.u = ttf1
            self.x = kx
            self.y = ky
            self.X, self.Y = ndgrid(self.x, self.y)

    def ifft(self, z=10 * mm, shift=True, remove0=True, matrix=False):
        """Fast Fourier Transform (fft) of the field.

        Parameters:
            z (float): distance to the observation plane or focal of lens
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. If False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else puts in self

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        """

        ttf1 = ifft2(self.u)

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = fftshift(ttf1)

        if matrix is True:
            return ttf1
        else:
            u3 = Scalar_field_XY(self.x, self.y, self.wavelength)

            # x scaling - Infor
            num_x = self.x.size
            delta_x = self.x[1] - self.x[0]
            freq_nyquist_x = 1 / (2 * delta_x)
            kx = linspace(-freq_nyquist_x, freq_nyquist_x,
                          num_x) * self.wavelength
            num_y = self.y.size
            delta_y = self.y[1] - self.y[0]
            freq_nyquist_y = 1 / (2 * delta_y)
            ky = linspace(-freq_nyquist_y, freq_nyquist_y,
                          num_y) * self.wavelength

            u3.x = kx
            u3.y = ky

            u3.X, u3.Y = ndgrid(u3.x, u3.y)

            if z is None:
                u3.x = kx  # exit in angles (radians)
            else:
                u3.x = kx * z  # exit distances at a obsrvation plane z

            u3.u = ttf1

            return u3

    def _RS_(self,
             z,
             n,
             new_field=True,
             out_matrix=False,
             kind='z',
             xout=None,
             yout=None,
             verbose=False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)` Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Parameters:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced

            xout (float), init point for amplification at x
            yout (float), init point for amplification at y
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X
            else None

        Note:
            One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.


        References:
             Applied Optics vol 45 num 6 pp. 1102-1110 (2006)

        """

        if xout is None:
            xout = self.x[0]
        if yout is None:
            yout = self.y[0]

        xout = self.x + xout - self.x[0]
        yout = self.y + yout - self.y[0]

        nx = len(xout)
        ny = len(yout)
        dx = xout[1] - xout[0]
        dy = yout[1] - yout[0]

        # parametro de quality
        dr_real = sqrt(dx**2 + dy**2)
        rmax = sqrt((xout**2).max() + (yout**2).max())
        dr_ideal = sqrt((self.wavelength / n)**2 + rmax**2 + 2 *
                        (self.wavelength / n) * sqrt(rmax**2 + z**2)) - rmax
        self.quality = dr_ideal / dr_real
        if verbose is True:
            if (self.quality.min() > 1):
                print('Good result: factor {:2.2f}'.format(self.quality))
            else:
                print('- Needs denser sampling: factor {:2.2f}'.format(
                    self.quality))

        a = [2, 4]
        num_repx = int(round((nx) / 2) - 1)
        num_repy = int(round((ny) / 2) - 1)
        bx = array(a * num_repx)
        by = array(a * num_repy)
        cx = concatenate(((1, ), bx, (2, 1))) / 3.
        cy = concatenate(((1, ), by, (2, 1))) / 3.

        if float(nx) / 2 == round(nx / 2):  # es par
            i_centralx = num_repx + 1
            cx = concatenate((cx[:i_centralx], cx[i_centralx + 1:]))
        if float(ny) / 2 == round(ny / 2):  # es par
            i_centraly = num_repy + 1
            cy = concatenate((cy[:i_centraly], cy[i_centraly + 1:]))

        d1x = matrix(cx)
        d1y = matrix(cy)
        W = array(d1y.T * d1x)

        U = zeros((2 * ny - 1, 2 * nx - 1), dtype=complex)
        U[0:ny, 0:nx] = array(W * self.u)

        xext = self.x[0] - xout[::-1]
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xout[0]))

        yext = self.y[0] - yout[::-1]
        yext = yext[0:-1]
        yext = concatenate((yext, self.y - yout[0]))

        Xext, Yext = meshgrid(xext, yext)

        # permite calcula la propagacion y la propagacion inverse, cuando z<0.
        if z > 0:
            H = kernelRS(Xext, Yext, self.wavelength, z, n, kind=kind)
        else:
            H = kernelRSinverse(Xext, Yext, self.wavelength, z, n, kind=kind)

        # calculo de la transformada de Fourier
        S = ifft2(fft2(U) * fft2(H)) * dx * dy
        # transpose cambiado porque daba problemas para matrices no cuadradas
        Usalida = S[ny - 1:, nx - 1:]  # hasta el final
        # los calculos se pueden dejar en la instancia o crear un new field

        # Usalida = Usalida / z  210131

        if out_matrix is True:
            return Usalida

        if new_field is True:
            field_output = Scalar_field_XY(self.x, self.y, self.wavelength)
            field_output.u = Usalida
            field_output.quality = self.quality
            return field_output
        else:
            self.u = Usalida

    def RS(self,
           z,
           amplification=(1, 1),
           n=1,
           new_field=True,
           matrix=False,
           kind='z',
           verbose=False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Parameters:
            amplification (int, int): number of frames in x and y direction
            z (float): distance to observation plane. if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u, if True a new instance is produced
            kind (str):
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X, else None.

        Note:
            One advantage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        References:
            Applied Optics vol 45 num 6 pp. 1102-1110 (2006).
        """

        amplification_x, amplification_y = amplification

        if amplification_x * amplification_y > 1:

            ancho_x = self.x[-1] - self.x[0]
            ancho_y = self.y[-1] - self.y[0]
            num_pixels_x = len(self.x)
            num_pixels_y = len(self.y)

            posiciones_x = -amplification_x * ancho_x / 2 + array(
                list(range(amplification_x))) * ancho_x
            posiciones_y = -amplification_y * ancho_y / 2 + array(
                list(range(amplification_y))) * ancho_y

            X0 = linspace(-amplification_x * ancho_x / 2,
                          amplification_x * ancho_x / 2,
                          num_pixels_x * amplification_x)
            Y0 = linspace(-amplification_y * ancho_y / 2,
                          amplification_y * ancho_y / 2,
                          num_pixels_y * amplification_y)

            U_final = Scalar_field_XY(x=X0, y=Y0, wavelength=self.wavelength)

            # TODO: pass to multiprocessing
            for i, xi in zip(list(range(len(posiciones_x))),
                             flipud(posiciones_x)):
                for j, yi in zip(list(range(len(posiciones_y))),
                                 flipud(posiciones_y)):
                    # num_ventana = j * amplification_x + i + 1
                    u3 = self._RS_(z=z,
                                   n=n,
                                   new_field=False,
                                   kind=kind,
                                   xout=xi,
                                   yout=yi,
                                   out_matrix=True,
                                   verbose=verbose)
                    xshape = slice(i * num_pixels_x, (i + 1) * num_pixels_x)
                    yshape = slice(j * num_pixels_y, (j + 1) * num_pixels_y)
                    U_final.u[yshape, xshape] = u3

            if matrix is True:
                return U_final.u
            else:
                if new_field is True:
                    return U_final
                else:
                    self.u = U_final.u
                    self.x = X0
                    self.y = Y0
        else:
            u_s = self._RS_(z,
                            n,
                            new_field=new_field,
                            out_matrix=True,
                            kind=kind,
                            xout=None,
                            yout=None,
                            verbose=verbose)

            if matrix is True:
                return u_s
            else:
                if new_field is True:
                    U_final = Scalar_field_XY(x=self.x,
                                              y=self.y,
                                              wavelength=self.wavelength)
                    U_final.u = u_s
                    return U_final
                else:
                    self.u = u_s

    def WPM(self,
            fn,
            z_ini,
            z_end,
            dz,
            has_edges=True,
            matrix=False,
            verbose=False):
        """
        WPM Methods.
        'schmidt method is very fast, only needs discrete number of refraction indexes'


        Parameters:
            fn - índice de refracción función:
            kind (str): 'schmidt, scalar, TE, TM
            filter (1, or np.array): filter for edges
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """
        z_now = z_ini
        k0 = 2 * np.pi / self.wavelength
        x = self.x
        y = self.y
        # dx = x[1] - x[0]
        # dy = y[1] - y[0]

        u_iter = Scalar_field_XY(self.x, self.y, self.wavelength)

        kx = get_k(x, flavour='+')
        ky = get_k(y, flavour='+')

        KX, KY = np.meshgrid(kx, ky)

        k_perp2 = KX**2 + KY**2
        # k_perp = np.sqrt(k_perp2)

        num_steps = int(z_end / dz)

        if has_edges is False:
            filter_edge = 1
        else:
            gaussX = np.exp(-(self.X / (self.x[0]))**86)
            gaussY = np.exp(-(self.Y / (self.y[0]))**86)
            filter_edge = (gaussX * gaussY)

        t1 = time.time()

        u_iter.u = self.u
        for j in range(0, num_steps):
            refraction_index = fn(x, y, np.array([
                z_now,
            ]), self.wavelength)

            u_iter.u = WPM_schmidt_kernel(u_iter.u, refraction_index, k0,
                                          k_perp2, dz) * filter_edge

            z_now = z_now + dz
            if verbose is True:
                if sys.version_info.major == 3:
                    print("{}/{}".format(j, num_steps), sep='\r', end='\r')
                else:
                    print("{}/{}".format(j, num_steps))

        t2 = time.time()
        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / num_steps * 1000))

        if matrix is True:
            return u_iter.u
        else:
            self.u = u_iter.u

    def profile(self,
                point1='',
                point2='',
                npixels=None,
                kind='intensity',
                order=2):
        """Determine profile in image. If points are not given, then image is shown and points are obtained clicking.

        Parameters:
            point1 (float): initial point. if '' get from click
            point2 (float): final point. if '' get from click
            npixels (int): number of pixels for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase'
            order (int): order for interpolation

        Returns:
            numpy.array: profile
            numpy.array: z values for profile
            (float, float): point1
            (float, float): point2
        """

        if npixels is None:
            npixels = len(self.x)

        if point1 == '' or point2 == '':
            self.draw(kind=kind)
            print("coordinates to given: click twice")
            point1, point2 = plt.ginput(2)

        x1, y1 = point1
        x2, y2 = point2

        ix1, value, distance = nearest(self.x, x1)
        ix2, value, distance = nearest(self.x, x2)
        iy1, value, distance = nearest(self.y, y1)
        iy2, value, distance = nearest(self.y, y2)

        x = linspace(ix1, ix2, npixels)
        y = linspace(iy1, iy2, npixels)

        if kind == 'intensity':
            image = np.abs(self.u)**2
        elif kind == 'amplitude':
            image = real(self.u)
        elif kind == 'phase':
            image = angle(self.u)  # / pi
            image[image == 1] = -1

        h = linspace(0, sqrt((y2 - y1)**2 + (x2 - x1)**2), npixels)
        h = h - h[-1] / 2

        z_profile = scipy.ndimage.map_coordinates(image.transpose(),
                                                  np.vstack((x, y)),
                                                  order=order)

        return h, z_profile, point1, point2

    def draw_profile(self,
                     point1='',
                     point2='',
                     npixels=None,
                     kind='intensity',
                     order=2):
        """Draws profile in image. If points are not given, then image is shown and points are obtained clicking.

        Parameters:
            point1 (float): initial point. if '' get from click
            point2 (float): final point. if '' get from click
            npixels (int): number of pixels for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase'
            order (int): order for interpolation

        Returns:
            numpy.array: profile
            numpy.array: z values for profile
            (float, float): point1
            (float, float): point2
        """

        if npixels is None:
            npixels = len(self.x)

        h, z_profile, point1, point2 = self.profile(point1, point2, npixels,
                                                    kind, order)

        plt.figure()
        plt.plot(h, z_profile, 'k', lw=2)
        plt.xlabel('h (profile)')
        plt.ylabel(kind)
        plt.axis([h.min(), h.max(), z_profile.min(), z_profile.max()])
        return h, z_profile, point1, point2

    def get_edges(self,
                  kind_transition='amplitude',
                  min_step=0,
                  verbose=False,
                  filename=''):
        """
        Determine locations of edges for a binary mask. Valid for litography engraving of gratings.

        Parameters:
            kind_transition:'amplitude' 'phase'.
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

    def search_focus(self, kind='moments', verbose=True):
        """Search for location of .

        Parameters:
            kind (str): 'moments' or 'maximum'
            verbose (bool): If True prints information.

        Returns:
            (x,y): positions of focus
        """

        if kind == 'maximum':
            intensity = np.abs(self.u)**2
            ix, iy = np.unravel_index(intensity.argmax(), intensity.shape)
            pos_x, pos_y = self.x[ix], self.y[iy]
        elif kind == 'moments':
            _, _, _, moments = beam_width_2D(self.x,
                                             self.y,
                                             np.abs(self.u)**2,
                                             has_draw=False)
            pos_x, pos_y, _, _, _ = moments

        if verbose is True:
            print(("x = {:2.3f} um, y = {:2.3f} um".format(pos_x, pos_y)))

        return pos_x, pos_y

    def MTF(self, kind='mm', has_draw=True, is_matrix=True):
        """Computes the MTF of a field, If this field is near to focal point, the MTF will be wide

        Parameters:
            kind (str): 'mm', 'degrees'
            has_draw (bool): If True draws the MTF

        Returns:
            (numpy.array) fx: frequencies in lines/mm
            (numpy.array) mtf_norm: normalizd MTF
        """

        tmp_field = self.u
        x = self.x
        y = self.y
        self.u = np.abs(self.u)**2
        MTF_field = self.fft(new_field=True, shift=True, remove0=False)

        num_data_x, num_data_y = MTF_field.u.shape

        mtf_norm = np.abs(MTF_field.u) / np.abs(
            MTF_field.u[int(num_data_x / 2),
                        int(num_data_y / 2)])

        delta_x = x[1] - x[0]
        delta_y = y[1] - y[0]

        frec_nyquist_x = 0.5 / delta_x
        frec_nyquist_y = 0.5 / delta_y

        fx = 1000 * np.linspace(-frec_nyquist_x, frec_nyquist_x, len(x))
        fy = 1000 * np.linspace(-frec_nyquist_y, frec_nyquist_y, len(y))

        if kind == 'mm':
            # frec_x = fx
            # frec_y = fy
            text_x = "$f_x (cycles/mm)$"
            text_y = "$f_y (cycles/mm)$"
        elif kind == 'degrees':
            print("not implemented yet")
            # frec_x = fx
            # frec_y = fy
            text_x = "$f_x (cycles/deg - not yet)$"
            text_y = "$f_x (cycles/deg - not yet)$"

        if has_draw is True:
            draw2D(
                mtf_norm,
                x,
                y,
                xlabel=text_x,
                ylabel=text_y,
                title="",
                color="gist_heat",  # YlGnBu  seismic
                interpolation='bilinear',  # 'bilinear', 'nearest'
                scale='scaled')
            plt.colorbar(orientation='vertical')

        self.u = tmp_field

        if is_matrix is True:
            return fx, fy, mtf_norm
        else:
            u_mtf = Scalar_field_XY(fx, fy, self.wavelength)
            u_mtf.u = mtf_norm
            return u_mtf

    def beam_width_4s(self, has_draw=True):
        """Returns the beam width parameters according to ISO11146.

        Parameters:
            has_draw (bool): If True, it draws

        Returns:
            (float): dx width x
            (float): dy width y
            (float): principal_axis, angle
            (str): (x_mean, y_mean, x2_mean, y2_mean, xy_mean), Moments

        References:

            * https://en.wikipedia.org/wiki/Beam_diameter
            * http://www.auniontech.com/ueditor/file/20170921/1505982360689799.pdf
    """
        dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean,
                                 xy_mean) = beam_width_2D(self.x,
                                                          self.y,
                                                          np.abs(self.u)**2,
                                                          has_draw=False)

        if has_draw is True:
            from matplotlib.patches import Ellipse

            self.draw()
            ellipse = Ellipse(xy=(x_mean, y_mean),
                              width=dy,
                              height=dx,
                              angle=-principal_axis / degrees)
            ellipse2 = Ellipse(xy=(x_mean, y_mean),
                               width=dy / 2,
                               height=dx / 2,
                               angle=-principal_axis / degrees)

            ellipse3 = Ellipse(xy=(x_mean, y_mean),
                               width=dy / 4,
                               height=dx / 4,
                               angle=-principal_axis / degrees)

            ax = plt.gca()
            ax.add_artist(ellipse)
            ellipse.set_clip_box(ax.bbox)
            ellipse.set_facecolor('none')
            ellipse.set_alpha(0.75)
            ellipse.set_edgecolor('yellow')
            ellipse.set_linewidth(3)

            ax.add_artist(ellipse2)
            ellipse2.set_clip_box(ax.bbox)
            ellipse2.set_facecolor('none')
            ellipse2.set_alpha(0.75)
            ellipse2.set_edgecolor('red')
            ellipse2.set_linewidth(3)

            ax.add_artist(ellipse3)
            ellipse3.set_clip_box(ax.bbox)
            ellipse3.set_facecolor('none')
            ellipse3.set_alpha(0.75)
            ellipse3.set_edgecolor('black')
            ellipse3.set_linewidth(3)

            x0 = self.x[0]
            y0 = self.y[0]
            plt.plot(x0, y0, 'yellow', label='4$\sigma$')
            plt.plot(x0, y0, 'red', label='2$\sigma$')
            plt.plot(x0, y0, 'black', label='1$\sigma$')
            plt.legend()

        return dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean,
                                        xy_mean)

    def intensity(self):
        """Returns intensity."""

        intensity = (np.abs(self.u)**2)
        return intensity

    def average_intensity(self, verbose=False):
        """Returns average intensity as: (np.abs(self.u)**2).sum() / num_data.

        Parameters:
            verbose(bool): If True prints data.
        """
        num_data = len(self.x) * len(self.y)
        average_intensity = (np.abs(self.u)**2).sum() / num_data
        if verbose is True:
            print(("average intensity={} W/m").format(average_intensity))

        return average_intensity

    def send_image_screen(self, id_screen, kind='amplitude'):
        """Takes the images and sends the images to a screen in full size.

        Parameters:
            id_screen(hdl): handle to screen
            kind('str'): 'amplitude', 'intensity', 'phase'
        """

        amplitude, intensity, phase = field_parameters(self.u)

        if kind == 'amplitude':
            image = amplitude
        elif kind == 'intensity':
            image = intensity
        elif kind == 'phase':

            phase = (phase + np.pi) % (2 * np.pi) - np.pi

            image = phase + np.pi
            image[0, 0] = 0
            image[0, 1] = 2 * np.pi
            image = image / (2 * np.pi)

        print(("send_image_screen: max={}. min={}".format(
            image.max(), image.min())))

        screen = screeninfo.get_monitors()[id_screen]
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def get_amplitude(self, matrix=False, new_field=False):
        """Gets the amplitude of the field.

        Parameters:
            matrix(bool): if True numpy.matrix is returned
            new_field(bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X
            if Matrix is True: numpy.array
        """
        amplitude = abs(self.u)

        if matrix is True:
            return amplitude

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = amplitude
            return u_salida

        else:
            self.u = amplitude

    def get_phase(self, matrix=False, new_field=False):
        """Gets the phase of the field.

        Parameters:
            matrix(bool): if True numpy.matrix is returned
            new_field(bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X.
            if Matrix is True: numpy.array.
        """
        phase = exp(1j * angle(self.u))

        if matrix is True:
            return phase

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = phase
            return u_salida

        else:
            self.u = phase

    def remove_phase(self, sign=False, matrix=False, new_field=False):
        """Removes the phase of the field. Amplitude is kept.

        Parameters:
            sign(bool): If True, sign is kept, else, it is removed
            matrix(bool): if True numpy.matrix is returned
            new_field(bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X.
            if Matrix is True: numpy.array.
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        if sign is False:
            only_amplitude = amplitude
        elif sign is True:
            only_amplitude = np.sign(phase) * amplitude

        if matrix is True:
            return only_amplitude

        if new_field is True:
            u_salida = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_salida.u = only_amplitude
            return u_salida

        else:
            self.u = only_amplitude

    def binarize(self,
                 kind="amplitude",
                 corte=None,
                 level0=None,
                 level1=None,
                 new_field=False,
                 matrix=False):
        """Changes the number of points in field, mantaining the area.

        Parameters:
            kind(str): 'amplitude' or 'phase'
            corte(float): value of cut. If None, the cut is in the mean value
            level0(float): minimum value. If None, minimum value of field
            level1(float): maximum value. If None, maximum value of field
            new_field(bool): if True returns new field
            matrix(bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO:
            Check and pass to utils
        """

        amplitude = self.get_amplitude(matrix=True, new_field=False)
        phase = self.get_phase(matrix=True, new_field=False)

        if kind == 'amplitude':
            amplitude_binarized = amplitude
            maximum = amplitude.max()
            minimum = amplitude.min()
            if corte is None:
                corte = (maximum + minimum) / 2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            amplitude_binarized[amplitude <= corte] = level0
            amplitude_binarized[amplitude > corte] = level1
            fieldDiscretizado = amplitude_binarized * phase

        if kind == 'phase':
            # phaseInicial = 0
            phase_binarized = phase
            maximum = phase.max()
            minimum = phase.min()
            if corte is None:
                corte = (maximum + minimum) / 2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            phase_binarized[phase <= corte] = level0
            phase_binarized[phase > corte] = level1
            fieldDiscretizado = amplitude * phase_binarized

        if new_field is False and matrix is False:
            self.u = fieldDiscretizado
            return self.u

        if new_field is False and matrix is True:
            return fieldDiscretizado

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = fieldDiscretizado
            return cn

    def discretize(self,
                   kind='amplitude',
                   num_levels=2,
                   factor=1,
                   phaseInicial=0,
                   new_field=True,
                   matrix=False):
        """Discretize in a number of levels equal to num_levels.

        Parameters:
            kind(str): "amplitude" o "phase"
            num_levels(int): number of levels for the discretization
            factor(float): from the level, how area is binarized. if 1 everything is binarized,
            phaseInicial(float): *
            new_field(bool): if True returns new field
            matrix(bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO:
            Check and pass to utils
        """

        if kind == 'amplitude':
            heights = linspace(0, 1, num_levels)
            posX = 256 / num_levels

            amplitude = self.get_amplitude(matrix=True, new_field=False)
            phase = self.get_phase(matrix=True, new_field=False)
            discretized_image = amplitude

            dist = factor * posX

            for i in range(num_levels):
                centro = posX / 2 + i * posX
                abajo = amplitude * 256 > centro - dist / 2
                arriba = amplitude * 256 <= centro + dist / 2
                Trues = abajo * arriba
                discretized_image[Trues] = centro / 256

            fieldDiscretizado = discretized_image * phase

        if kind == 'phase':
            ang = angle(self.get_phase(matrix=True,
                                       new_field=False)) + phaseInicial + pi
            ang = ang % (2 * pi)
            amplitude = self.get_amplitude(matrix=True, new_field=False)

            heights = linspace(0, 2 * pi, num_levels + 1)

            dist = factor * (heights[1] - heights[0])

            discretized_image = exp(1j * (ang))

            for i in range(num_levels + 1):
                centro = heights[i]
                abajo = (ang) > (centro - dist / 2)
                arriba = (ang) <= (centro + dist / 2)
                Trues = abajo * arriba
                discretized_image[Trues] = exp(1j * (centro))  # - pi

            Trues = (ang) > (centro + dist / 2)
            discretized_image[Trues] = exp(1j * (heights[0]))  # - pi

            phase = angle(discretized_image) / pi
            phase[phase == 1] = -1
            phase = phase - phase.min()
            discretized_image = exp(1j * pi * phase)

            fieldDiscretizado = amplitude * discretized_image

        if new_field is False and matrix is False:
            self.u = fieldDiscretizado
            return

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = fieldDiscretizado
            return cn

        if matrix is True:
            return fieldDiscretizado

    def normalize(self, kind='intensity'):
        """Normalize the field.

        Parameters:
            kind(str): 'intensity' 'area'

        TODO:
            pass to utils  # esto lo he puesto a última hora
        """

        if kind == 'intensity':
            intensity = abs(self.u**2)
            maximum = sqrt(intensity.max())
            self.u = self.u / maximum
        if kind == 'area':
            intensity = abs(self.u**2)
            maximum = intensity.sum()
            self.u = self.u / maximum

    def draw(self,
             kind='intensity',
             logarithm=False,
             normalize=False,
             title="",
             filename='',
             cut_value=None,
             has_colorbar='',
             colormap_kind='',
             reduce_matrix='standard',
             **kwargs):
        """Draws  XY field.

        Parameters:
            kind(str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'contour'
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            filename(str): if not '' stores drawing in file,
            cut_value(float): if provided, maximum value to show
            has_colorbar(bool): if True draws the colorbar
            reduce_matrix(str): 'standard'
        """

        if reduce_matrix in ([], None, ''):
            pass
        else:
            self.reduce_matrix = reduce_matrix

        if kind == 'intensity':
            id_fig, IDax, IDimage = self.__draw_intensity__(
                logarithm, normalize, title, cut_value, colormap_kind,
                **kwargs)
        elif kind == 'amplitude':
            id_fig, IDax, IDimage = self.__drawAmplitude__(
                logarithm, normalize, title, cut_value, colormap_kind,
                **kwargs)
        elif kind == 'phase':
            id_fig, IDax, IDimage = self.__draw_phase__(
                title, colormap_kind, **kwargs)
        elif kind == 'field':
            id_fig = self.__draw_field__(logarithm, normalize, title,
                                         cut_value, colormap_kind, **kwargs)
            IDax = None
            IDimage = None
        elif kind == 'real_field':
            id_fig, IDax, IDimage = self.__draw_real_field__(
                logarithm, normalize, title, cut_value, **kwargs)
        else:
            print("not in kinds")

        if has_colorbar in ('horizontal', 'vertical'):
            plt.colorbar(orientation=has_colorbar, shrink=0.75)

        if not filename == '':
            plt.savefig(filename,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.05)

        return id_fig, IDax, IDimage

    def __draw_intensity__(self,
                           logarithm=False,
                           normalize='maximum',
                           title="",
                           cut_value=None,
                           colormap_kind='',
                           **kwargs):
        """Draws intensity  XY field.

        Parameters:
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        if colormap_kind in ['', None, []]:
            colormap_kind = self.params_drawing["color_intensity"]
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)
        id_fig, IDax, IDimage = draw2D(intensity,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix, **kwargs)

        if self.type == 'Scalar_mask_XY':
            plt.clim(0, 1)

        return id_fig, IDax, IDimage

    def __drawAmplitude__(self,
                          logarithm=False,
                          normalize='maximum',
                          title='intensity',
                          cut_value=1,
                          colormap_kind=''):
        """Draws amplitude  XY field.

        Parameters:
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        amplitude = normalize_draw(amplitude, logarithm, normalize, cut_value)
        max_amplitude = np.abs(amplitude).max()
        if colormap_kind in ['', None, []]:
            colormap_kind = self.params_drawing["color_amplitude"]
        id_fig, IDax, IDimage = draw2D(amplitude,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix)
        plt.clim(-max_amplitude, max_amplitude)

        return id_fig, IDax, IDimage

    def __draw_phase__(self, title=r'phase/pi', colormap_kind=''):
        """Draws phase of  XY field

        Parameters:
            title(str): title for the drawing
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        phase[phase == 1] = -1
        phase = phase / degrees
        phase[intensity < percentaje_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.params_drawing["color_phase"]

        id_fig, IDax, IDimage = draw2D(
            phase,
            self.x,
            self.y,
            xlabel="$x  (\mu m)$",
            ylabel="$y  (\mu m)$",
            title=title,
            color=colormap_kind,
            reduce_matrix=self.reduce_matrix)  # seismic gist_heat
        plt.clim(vmin=-180, vmax=180)

        return id_fig, IDax, IDimage

    def __draw_field__(self,
                       logarithm=False,
                       normalize='maximum',
                       title="",
                       cut_value=None,
                       colormap_kind='',
                       **kwargs):
        """Draws field  XY field.

        Parameters:
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """

        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.y,
                                       intensity)

        phase = reduce_matrix_size(self.reduce_matrix, self.x, self.y, phase)
        phase[intensity < percentaje_intensity * (intensity.max())] = 0

        xsize, ysize = rcParams['figure.figsize']

        plt.figure(figsize=(2 * xsize, ysize))
        plt.suptitle(title)
        extension = [self.x[0], self.x[-1], self.y[0], self.y[-1]]

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.subplot(1, 2, 1)

        h1 = plt.imshow(intensity,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel("$x  (\mu m)$")
        plt.ylabel("$y  (\mu m)$")
        plt.title("$intensity$")
        plt.axis('scaled')
        plt.axis(extension)
        plt.colorbar(orientation='horizontal', shrink=0.66)
        plt.axis(extension)
        h1.set_cmap(self.params_drawing["color_intensity"])
        if self.type == 'Scalar_mask_XY':
            plt.clim(0, 1)

        plt.subplot(1, 2, 2)
        phase[phase == 1] = -1
        phase = phase / degrees

        # elimino la fase en la visualicion cuando no hay campo
        h2 = plt.imshow(phase,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel("$x  (\mu m)$")
        plt.ylabel("$y  (\mu m)$")
        plt.colorbar(orientation='horizontal', shrink=0.66)
        plt.axis('scaled')
        plt.axis(extension)
        plt.title("$phase$")
        plt.clim(-180, 180)
        h2.set_cmap(self.params_drawing["color_phase"])  #
        plt.subplots_adjust(0, 0, 1, 1, 0, 0)

        return (h1, h2)

    def __draw_real_field__(self,
                            logarithm=False,
                            normalize='maximum',
                            cut_value=1,
                            title="",
                            colormap_kind=''):
        """Draws real field  XY field.

        Parameters:
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """

        rf = np.real(self.u)
        intensity = np.abs(self.u)**2
        rf[intensity < percentaje_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.params_drawing["color_real"]

        id_fig, IDax, IDimage = draw2D(rf,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix)

        return id_fig, IDax, IDimage

    def video(self,
              kind,
              zs,
              logarithm=False,
              normalize=False,
              time_video=10 * seconds,
              frames_reduction=1,
              filename='video.avi',
              dpi=300):
        """Makes a video

        Parameters:
            kind(str): 'intensity', 'phase', 'amplitude'
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.grid()
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(self.y[0], self.y[-1])

        def animate(i):
            t2 = self.RS(z=zs[i], new_field=True)

            image = reduce_matrix_size(self.reduce_matrix, self.x, self.y,
                                       t2.u)

            print(("image size {}".format(image.shape)))

            I_drawing = prepare_drawing(image, kind, logarithm, normalize)
            ax.imshow(I_drawing)
            ax.set_title("$z = {:2.0f} \mu m$".format(zs[i]))
            return i

        ani = animation.FuncAnimation(fig,
                                      animate,
                                      list(range(0, len(zs),
                                                 frames_reduction)),
                                      interval=25,
                                      blit=False)

        fps = int(len(zs) / (time_video * frames_reduction))

        ani.save(filename, fps=fps, dpi=dpi)


def kernelRS(X, Y, wavelength, z, n=1, kind='z'):
    """Kernel for RS propagation.

    Había una errata según: Juan Valencia <valen167@umn.edu>  210131

    Parameters:
        X(numpy.array): positions x
        Y(numpy.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refraction index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """
    k = 2 * pi * n / wavelength
    R = sqrt(X**2 + Y**2 + z**2)
    if kind == 'z':
        return 1 / (2 * pi) * exp(1.j * k * R) * z / R**2 * (1 / R - 1.j * k)
    elif kind == 'x':
        return 1 / (2 * pi) * exp(1.j * k * R) * X / R**2 * (1 / R - 1.j * k)
    elif kind == 'y':
        return 1 / (2 * pi) * exp(1.j * k * R) * Y / R**2 * (1 / R - 1.j * k)
    elif kind == '0':
        return 1 / (2 * pi) * exp(1.j * k * R) / R * (1 / R - 1.j * k)


def kernelRSinverse(X, Y, wavelength=0.6328 * um, z=-10 * mm, n=1, kind='z'):
    """Kernel for inverse RS propagation

    Parameters:
        X(numpy.array): positions x
        Y(numpy.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refraction index of background
        kind(str): 'z', 'x', '0': for simplifying vector propagation

    Returns:
        complex np.array: kernel
    """
    k = 2 * pi * n / wavelength
    R = sqrt(X**2 + Y**2 + z**2)
    if kind == 'z':
        return 1 / (2 * pi) * exp(-1.j * k * R) * z / R * (1 / R + 1.j * k)
    elif kind == 'x':
        return 1 / (2 * pi) * exp(-1.j * k * R) * X / R * (1 / R + 1.j * k)
    elif kind == 'y':
        return 1 / (2 * pi) * exp(-1.j * k * R) * Y / R * (1 / R + 1.j * k)
    elif kind == '0':
        return 1 / (2 * pi) * exp(-1.j * k * R) * (1 / R + 1.j * k)


def kernelFresnel(X, Y, wavelength=0.6328 * um, z=10 * mm, n=1):
    """Kernel for Fesnel propagation

    Parameters:
        X(numpy.array): positions x
        Y(numpy.array): positions y
        wavelength(float): wavelength of incident fields
        z(float): distance for propagation
        n(float): refraction index of background

    Returns:
        complex np.array: kernel
    """
    k = 2 * pi * n / wavelength
    return exp(1.j * k * (z + (X**2 + Y**2) /
                          (2 * z))) / (1.j * wavelength * z)


def PWD_kernel(u, n, k0, k_perp2, dz):
    """
    Step for scalar(TE) Plane wave decomposition(PWD) algorithm.

    Parameters:
        u(np.array): field
        n(np.array): refraction index
        k0(float): wavenumber
        k_perp(np.array): transversal k
        dz(float): increment in distances

    Returns:
        (numpy.array): Field at at distance dz from the incident field

    References:
        1. Schmidt, S. et al. Wave - optical modeling beyond the thin - element - approximation. Opt. Express 24, 30188 (2016).

    """
    absorption = 0.00

    Ek = fftshift(fft2(u))
    H = np.exp(1j * dz * csqrt(n**2 * k0**2 - k_perp2) - absorption)

    result = (ifft2(fftshift(H * Ek)))
    return result


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

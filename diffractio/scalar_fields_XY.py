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
    * instatiation, duplicate
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
    * kernelRS, kernelRSinverse, kernelFresnel, WPM_schmidt_kernel
"""

import copy
import datetime
import time

import matplotlib.animation as animation
import scipy.ndimage
from matplotlib import rcParams
from numpy import (angle, array, concatenate, cos, exp, flipud, linspace,
                   meshgrid, pi, real, shape, sin, sqrt, zeros)
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import RectBivariateSpline

from . import np, plt
from . import degrees, mm, seconds, um

from .config import CONF_DRAWING
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import (draw2D, normalize_draw, prepare_drawing,
                            reduce_matrix_size)
from .utils_math import (get_edges, get_k, ndgrid, nearest, nearest2,
                         reduce_to_1, rotate_image, Bluestein_dft_xy)
from .utils_optics import beam_width_2D, field_parameters, normalize_field
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_fields_Z import Scalar_field_Z

try:
    import screeninfo
except:
    print("screeninfo not imported.")

try:
    import cv2
except:
    print("cv2 not imported. Function send_image_screen cannot be used")

percentage_intensity_config = CONF_DRAWING['percentage_intensity']


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
        self.CONF_DRAWING = CONF_DRAWING

    def __str__(self):
        """Represents main data of the atributes"""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - x:  {},   y:  {},   u:  {}".format(
            self.type, self.x.shape, self.y.shape, self.u.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - ymin:       {:2.2f} um,  ymax:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ("")

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

        TODO:  It can be improved for masks (not having less than 1)
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

    def conjugate(self, new_field=True):
        """Conjugates the field
        """

        if new_field is True:
            u_new = self.duplicate()
            u_new.u = np.conj(self.u)
            return u_new
        else:
            self.u = np.conj(self.u)

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        # new_field = Scalar_field_XY(self.x, self.y, self.wavelength)
        # new_field.u = self.u
        # tipo = type(self)
        # new_field = tipo(self)
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def reduce_to_1(self):
        """All the values greater than 1 pass to 1. This is used for Scalar_masks when we add two masks.
        """

        self = reduce_to_1(self)

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

    def rotate(self, angle, position=None, new_field=False):
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

        if new_field is True:
            u_new = Scalar_field_XY(self.x, self.y, self.wavelength)
            u_new.u = u_rotate
            return u_new
        else:
            self.u = u_rotate

    def apodization(self, power=10):
        """Multiply field by an apodizer. The apodizer is a super_gauss function.

        Args:
            power (int, optional): size of apodization. Defaults to 10.


        """
        width_x = (self.x[-1] - self.x[0]) / 2
        width_y = (self.y[-1] - self.y[0]) / 2

        center_x = (self.x[-1] + self.x[0]) / 2
        center_y = (self.y[-1] + self.y[0]) / 2

        t = np.exp(-((self.X - center_x) / width_x)**power -
                   ((self.Y - center_y) / width_y)**power)
        self.u = t

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

        if verbose is True:
            print(dict0.keys())

    def save_mask(self,
                  filename="",
                  kind='amplitude',
                  binarize=False,
                  cmap='gray',
                  info=""):
        """Create a mask in a file, for example, ablation or litography engraver

        Parameters:
            filename (str): file name
            kind (str): save amplitude, phase or intensity. If intensity it is normalized to (0,1)
            binarize (bool): If True convert the mask in (0,1) levels
            cmap (str): colormap. (gray)
            info (str): info of the mask

        Returns:
            float: area (in um**2)
        """

        # filename for txt
        name = filename.split(".")
        nombreTxt = name[0] + ".txt"

        # image
        plt.figure()
        filter = np.abs(self.u) > 0

        if kind == 'amplitude':
            mask = np.abs(self.u)

        elif kind == 'phase':
            mask = np.angle(self.u)
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = mask * filter

        elif kind == 'intensity':
            mask = np.abs(self.u)**2
            cmap = 'hot'

        if binarize is True:
            mask_min = mask.min()
            mask_max = mask.max()
            mask_mean = (mask_max + mask_min) / 2

            mask2 = np.zeros_like(mask)
            mask2[mask < mask_mean] = 0
            mask2[mask >= mask_mean] = 1

            mask = mask2

        plt.imsave(filename, mask, cmap=cmap, dpi=100, origin='lower')
        plt.close()

        # important data
        ofile = open(nombreTxt, "w")
        ofile.write("filename %s\n" % filename)
        ofile.write("date: {}\n".format(datetime.date.today()))
        if info is not None:
            ofile.write("\ninfo:\n")
            ofile.write(info)
        ofile.write("\n\n")
        ofile.write("mask length: %i x %i\n" % (len(self.x), len(self.y)))
        ofile.write("x0 = %f *um, x1 = %f *um, Deltax = %f *um\n" %
                    (self.x.min(), self.x[-1], self.x[1] - self.x[0]))
        ofile.write("y0 = %f *um, y1 = %f *um, Deltay = %f *um\n" %
                    (self.y.min(), self.y[-1], self.y[1] - self.y[0]))

        ofile.write("\wavelength = %f *um" % self.wavelength)
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
            u_new_abs = f_interp_abs(y_new, x_new)
            u_new_phase = f_interp_phase(y_new, x_new)
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

    def pupil(self, r0=None, radius=None, angle=0 * degrees):
        """place a pupil in the field. If r0 or radius are None, they are computed using the x,y parameters.

        Parameters:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            pupil(r0=(0 * um, 0 * um), radius=(250 * \
                   um, 125 * um), angle=0 * degrees)
        """

        if r0 is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            y0 = (self.y[-1] + self.y[0]) / 2
            r0 = (x0, y0)

        if radius is None:
            radiusx = (self.x[-1] - self.x[0]) / 2
            radiusy = (self.y[-1] - self.y[0]) / 2
            radius = (radiusx, radiusy)

        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Definicion de la transmitancia
        pupil0 = zeros(shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + \
            (Yrot)**2 / (radiusy**2 + 1e-15) < 1
        pupil0[ipasa] = 1
        self.u = self.u * pupil0

    """
    def fft_proposal(self,
                     z=0,
                     shift=True,
                     remove0=True,
                     matrix=False,
                     new_field=False,
                     cut_square=True):
        Fast Fourier Transform (FFT) of the field.
        Parameters:
            z (float): distance to the observation plane or focal of lens
                       if z==0, no x,y scaled is produced
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self
            cut_square (bool): when the mask is not squared, the fft is computed using the squared XY version of the mask, removing the laterals.

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        
        k = 2 * np.pi / self.wavelength
        num_x = self.x.size
        num_y = self.y.size
        #print(num_x, num_y, np.bitwise_and(num_x != num_y, cut_square == True))
        u_field = self.u
        range_x = self.x[1] - self.x[0]
        range_y = self.y[1] - self.y[0]
        # if np.bitwise_and(num_x != num_y, cut_square == True):
        #     num_final = min(num_x, num_y)
        #     print(num_final)
        #     u_field = fftshift(u_field)
        #     u_field = u_field[0:num_final, 0:num_final]

        ttf1 = np.fft.fft2(u_field)
        # print(ttf1.shape)

        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
                         num_x) * self.wavelength

        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.y * num_x / num_y
            y_new = self.x
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = ndgrid(self.x, self.y)
    """

    def fft(self,
            z=0,
            shift=True,
            remove0=True,
            matrix=False,
            new_field=False):
        """Fast Fourier Transform (FFT) of the field.

        Parameters:
            z (float): distance to the observation plane or focal of lens
                       if z==0, no x,y scaled is produced
            shift (bool): if True, fftshift is performed
            remove0 (bool): if True, central point is removed
            matrix (bool):  if True only matrix is returned. if False, returns Scalar_field_X
            new_field (bool): if True returns Scalar_field_X, else it puts in self

        Returns:
            (np.array or Scalar_field_X or None): FFT of the input field
        """

        k = 2 * np.pi / self.wavelength

        ttf1 = np.fft.fft2(self.u)

        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
                         num_x) * self.wavelength

        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.x
            y_new = self.y
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = ndgrid(self.x, self.y)

    def ifft_proposal(self,
                      z=0 * mm,
                      shift=True,
                      remove0=True,
                      matrix=False,
                      new_field=False):
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
        k = 2 * np.pi / self.wavelength
        num_x = self.x.size
        num_y = self.y.size
        u_field = self.u

        range_x = self.x[1] - self.x[0]
        range_y = self.y[1] - self.y[0]

        ttf1 = np.fft.ifft2(u_field)

        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
                         num_x) * self.wavelength

        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.y
            y_new = self.x * num_x / num_y
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = ndgrid(self.x, self.y)

    def ifft(self,
             z=0 * mm,
             shift=True,
             remove0=True,
             matrix=False,
             new_field=False):
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
        k = 2 * np.pi / self.wavelength

        ttf1 = np.fft.ifft2(self.u)
        ttf1 = ttf1

        # * np.exp(-1j * k * (z + (self.X**2 +
        #                                      self.Y**2) / (2 * z))) / (-1j * self.wavelength * z)

        if remove0 is True:
            ttf1[0, 0] = 0

        if shift is True:
            ttf1 = np.fft.fftshift(ttf1)

        if matrix is True:
            return ttf1

        # x scaling - Infor
        num_x = self.x.size
        delta_x = self.x[1] - self.x[0]
        freq_nyquist_x = 1 / (2 * delta_x)
        kx = np.linspace(-freq_nyquist_x, freq_nyquist_x,
                         num_x) * self.wavelength

        num_y = self.y.size
        delta_y = self.y[1] - self.y[0]
        freq_nyquist_y = 1 / (2 * delta_y)
        ky = np.linspace(-freq_nyquist_y, freq_nyquist_y,
                         num_y) * self.wavelength

        if z is None:
            x_new = kx  # exit in angles (radians)
            y_new = ky  # exit in angles (radians)
        elif z == 0:
            x_new = self.x
            y_new = self.y
        else:
            x_new = kx * z  # exit distances at a observation plane z
            y_new = ky * z  # exit distances at a observation plane z

        if new_field is True:
            field_output = Scalar_field_XY(x_new, y_new, self.wavelength)
            field_output.u = ttf1
            return field_output
        else:
            self.u = ttf1
            self.x = x_new
            self.y = y_new
            self.X, self.Y = ndgrid(self.x, self.y)

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
            F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.

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

        dr_real = sqrt(dx**2 + dy**2)
        rmax = sqrt((xout**2).max() + (yout**2).max())
        dr_ideal = sqrt((self.wavelength / n)**2 + rmax**2 + 2 *
                        (self.wavelength / n) * sqrt(rmax**2 + z**2)) - rmax
        self.quality = dr_ideal / dr_real

        if verbose is True:
            if (self.quality.min() >= 0.99):
                print('Good result: factor {:2.2f}'.format(self.quality),
                      end='\r')
            else:
                print('- Needs denser sampling: factor {:2.2f}\n'.format(
                    self.quality))
        precise = 0
        if precise:
            a = [4, 2]
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

            W = (cx[:, np.newaxis] * cy[np.newaxis, :]).T

        else:
            W = 1

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
           xout=None,
           yout=None,
           kind='z',
           verbose=False):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula. Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        Parameters:
            amplification (int, int): number of frames in x and y direction
            z (float): distance to observation plane. if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u, if True a new instance is produced.
            matrix(Bool): if True returns a matrix, else a Scalar_field_XY
            xout (float): If not None, the sampling area is moved. This is the left position
            yout (float): If not None, the sampling area y moved. This is the lower position.
            kind (str):
            verbose (bool): if True it writes to shell

        Returns:
            if New_field is True: Scalar_field_X, else None.

        Note:
            One advantage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        References:
            F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.
        """

        amplification_x, amplification_y = amplification
        ancho_x = self.x[-1] - self.x[0]
        ancho_y = self.y[-1] - self.y[0]
        num_pixels_x = len(self.x)
        num_pixels_y = len(self.y)

        if amplification_x * amplification_y > 1:

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

            if xout is None:
                u_s = self._RS_(z,
                                n,
                                new_field=new_field,
                                out_matrix=True,
                                kind=kind,
                                xout=xout,
                                yout=yout,
                                verbose=verbose)
            else:
                print(self.x[0])
                print("\n")
                u_s = self._RS_(z,
                                n,
                                new_field=new_field,
                                out_matrix=True,
                                kind=kind,
                                xout=-xout + self.x[0] - ancho_x / 2,
                                yout=-yout + self.y[0] - ancho_y / 2,
                                verbose=verbose)

            if matrix is True:
                return u_s

            if new_field is True:
                U_final = Scalar_field_XY(x=self.x,
                                          y=self.y,
                                          wavelength=self.wavelength)
                U_final.u = u_s
                if xout is not None:
                    U_final.x = self.x + xout - self.x[0]
                    U_final.y = self.y + yout - self.y[0]
                    U_final.X, U_final.Y = meshgrid(self.x, self.y)

                return U_final
            else:
                self.u = u_s
                self.x = self.x + xout - self.x[0]
                self.y = self.y + yout - self.y[0]
                self.X, self.Y = meshgrid(self.x, self.y)

    def WPM(self,
            fn,
            zs,
            num_sampling=(512, 512),
            ROI=(None, None),
            r_pos=None,
            z_pos=None,
            get_u_max=False,
            has_edges=True,
            pow_edge=80,
            matrix=False,
            verbose=False):
        """WPM method used for very dense sampling. It does not storages the intensity distribution at propagation, but only selected areas. The areas to be stored are:
            - global view with a desired sampling given by num_sampling.
            - intensity at the last plane.
            - Intensity at plane with maximum intensity (focus of a lens, for example)
            - Region of interest given by ROI.

        Parameters:
            fn (function): Function that returns the refraction index at a plane in a numpy.array
            zs (np.array): linspace with positions z where to evaluate the field.
            num_sampling (int, int) or None: If None, it does not storage intermediate data. Otherwise, it stores a small XZ matrix which size in num_sampling.
            ROI (np.array, np.array) or None: x and z arrays with positions and sampling to store. It is used when we need to store with a high density a small area of the total field.
            r_pos (float or None): If not None, a high density longitudinal array with the field at x_pos position is stored.
            z_pos (float or None): If not None, a high density transversal array with the field at z_pos is stored.
            get_u_max (bool): If True, returns field at z position with maximum intensity.
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian.
            verbose (bool): If True prints information.

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """
        from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
        k0 = 2 * np.pi / self.wavelength
        x = self.x
        y = self.y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = zs[1] - zs[0]

        u_iter = Scalar_field_XY(self.x, self.y, self.wavelength)

        kx = get_k(x, flavour='+')
        ky = get_k(y, flavour='+')

        KX, KY = np.meshgrid(kx, ky)

        k_perp2 = KX**2 + KY**2
        k_perp = np.sqrt(k_perp2)

        if has_edges is False:
            filter_edge = 1
        else:
            gaussX = np.exp(-(self.X / (self.x[0]))**pow_edge)
            gaussY = np.exp(-(self.Y / (self.y[0]))**pow_edge)
            filter_edge = (gaussX * gaussY)

        u_iter = self.duplicate()

        # STORING field at maximum intensity
        if get_u_max is True:
            u_max = self.duplicate()
            I_max = 0.
            z_max = 0
        else:
            u_max = None
            z_max = None

        # Storing intensities at axis (x=0,0 de momento)
        intensities = np.zeros_like(zs)
        index_x_axis, _, _ = nearest(x, 0.)
        index_y_axis, _, _ = nearest(y, 0.)

        # Storing intensities at axis (x=x_pos)
        if r_pos is not None:
            from diffractio.scalar_fields_Z import Scalar_field_Z
            x_pos, y_pos = r_pos
            u_axis_x = Scalar_field_Z(zs, self.wavelength)
            index_x_axis, _, _ = nearest(x, x_pos)
            index_y_axis, _, _ = nearest(y, y_pos)
        else:
            u_axis_x = None

        # Storing fields at plane
        if z_pos is not None:
            u_axis_z = self.duplicate()
            index_zpos, _, _ = nearest(zs, z_pos)
        else:
            u_axis_z = None

        # STORING global view matrices
        if num_sampling is not None:

            len_x, len_y, len_z = num_sampling
            xout_gv = np.linspace(x[0], x[-1], len_x)
            yout_gv = np.linspace(y[0], y[-1], len_y)
            zout_gv = np.linspace(zs[0], zs[-1], len_z)
            u_out_gv = Scalar_field_XYZ(xout_gv,
                                        yout_gv,
                                        zout_gv,
                                        self.wavelength,
                                        info='from WPM_no_storage_2D')
            indexes_x_gv, _, _ = nearest2(x, xout_gv)
            indexes_y_gv, _, _ = nearest2(y, yout_gv)
            indexes_z_gv, _, _ = nearest2(zs, zout_gv)
            indexes_X_gv, indexes_Y_gv = np.meshgrid(indexes_x_gv,
                                                     indexes_y_gv)

            u_out_gv.n = fn(xout_gv, yout_gv, zout_gv, self.wavelength)
        else:
            u_out_gv = None

        # STORING ROI
        if ROI is not None:
            xout_roi, yout_roi, zout_roi = ROI
            u_out_roi = Scalar_field_XYZ(xout_roi,
                                         yout_roi,
                                         zout_roi,
                                         self.wavelength,
                                         info='from WPM_no_storage_2D')
            indexes_x_roi, _, _ = nearest2(x, xout_roi)
            indexes_y_roi, _, _ = nearest2(y, yout_roi)
            indexes_z_roi, _, _ = nearest2(zs, zout_roi)

            indexes_X_roi, indexes_Y_roi = np.meshgrid(indexes_x_roi,
                                                       indexes_y_roi)
            # indexes_x_roi = np.unique(indexes_x_roi)
            # indexes_y_roi = np.unique(indexes_y_roi)
            # indexes_z_roi = np.unique(indexes_z_roi)

            u_out_roi.n = fn(xout_roi, yout_roi, zout_roi, self.wavelength)
        else:
            u_out_roi = None

        t1 = time.time()
        num_steps = len(zs)
        iz_out_gv = 0
        iz_out_roi = 0

        for j in range(1, num_steps):
            refraction_index = fn(x, y, np.array([
                zs[j - 1],
            ]), self.wavelength)

            u_iter.u = WPM_schmidt_kernel(u_iter.u, refraction_index, k0,
                                          k_perp2, dz) * filter_edge

            current_intensity = np.max(np.abs(u_iter.u)**2)
            intensities[j] = u_iter.intensity()[index_x_axis, index_y_axis]

            if r_pos is not None:
                u_axis_x.u[j] = u_iter.u[index_x_axis, index_y_axis]

            if z_pos is not None:
                if j == index_zpos:
                    u_axis_z.u = u_iter.u

            if num_sampling is not None:
                if j in indexes_z_gv:
                    u_out_gv.u[:, :, iz_out_gv] = u_iter.u[indexes_X_gv,
                                                           indexes_Y_gv]
                    iz_out_gv = iz_out_gv + 1

            if ROI is not None:
                if j in indexes_z_roi:
                    u_out_roi.u[:, :, iz_out_roi] = u_iter.u[indexes_X_roi,
                                                             indexes_Y_roi]
                    iz_out_roi = iz_out_roi + 1

            if get_u_max is True:
                current_intensity = np.max(np.abs(u_iter.u)**2)
                if current_intensity > I_max:
                    I_max = u_iter.intensity().max()
                    u_max.u = u_iter.u
                    z_max = zs[j]

            if verbose is True:
                print("{}/{}".format(j, num_steps), sep='\r', end='\r')

        t2 = time.time()

        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / num_steps * 1000))

        return u_iter, u_out_gv, u_out_roi, u_axis_x, u_axis_z, u_max, z_max

    def CZT_backup(self, z, xout=None, yout=None, verbose=False):
        """Chirped Z Transform algorithm for XY Scheme. z, xout, and yout parameters can be numbers or arrays.
        The output Scheme depends on this input parameters.

        Parameters:
            z (float): diffraction distance
            xout (np.array): x array with positions of the output plane
            yout (np.array): y array with positions of the output plane
            verbose (bool): If True, it prints some information

        Returns:
            u_out: Scalar_field_** depending of the input scheme. When all the parameters are numbers, it returns the complex field at that point.


        References:
             [Light: Science and Applications, 9(1), (2020)] 
        """

        if xout is None:
            xout = self.x

        if yout is None:
            yout = self.y

        k = 2 * np.pi / self.wavelength

        if isinstance(z, (float, int)):
            num_z = 1
            # print("z = 0 dim")
        else:
            num_z = len(z)
            # print("z = 1 dim")

        if isinstance(xout, (float, int)):
            num_x = 1
            # print("x = 0 dim")
            xstart = xout
            xend = xout
        else:
            num_x = len(xout)
            # print("x = 1 dim")

            xstart = xout[0]
            xend = xout[-1]

        if isinstance(yout, (float, int)):
            num_y = 1
            # print("y = 0 dim")
            ystart = yout
            yend = yout
        else:
            num_y = len(yout)
            # print("y = 1 dim")

            ystart = yout[0]
            yend = yout[-1]

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        delta_out = np.zeros(2)
        if num_x > 1:
            delta_out[0] = (xend - xstart) / (num_x - 1)

        if num_y > 1:
            delta_out[1] = (yend - ystart) / (num_y - 1)

        Xout, Yout = np.meshgrid(xout, yout)

        if verbose:
            print("num x, num y, num z = {}, {}, {}".format(
                num_x, num_y, num_z))

        if num_z == 1:
            # calculating scalar diffraction below
            # F0 = np.exp(1j * k * z) / (1j * self.wavelength * z) * np.exp(
            #     1j * k / 2 / z * (Xout**2 + Yout**2))
            # F = np.exp(1j * k / 2 / z * (self.X**2 + self.Y**2))

            R = np.sqrt(Xout**2 + Yout**2 + z**2)
            F0 = 1 / (2 * np.pi) * np.exp(
                1.j * k * R) * z / R**2 * (1 / R - 1.j * k)

            R = np.sqrt(self.X**2 + self.Y**2 + z**2)
            F = 1 / (2 * np.pi) * np.exp(
                1.j * k * R) * z / R**2 * (1 / R - 1.j * k)

            u0 = self.u * F

            # using Bluestein method to calculate the complex amplitude of the outgoing light beam

            # one-dimensional FFT in one direction
            fs = self.wavelength * z / dx  # dimension of the imaging plane

            if num_x > 1 and num_y == 1:

                # one-dimensional FFT in the other direction
                fx1 = xstart + fs / 2
                fx2 = xend + fs / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fs, num_x)

                fy1 = ystart + fs / 2
                fy2 = yend + fs / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fs, num_y)

            else:

                fy1 = ystart + fs / 2
                fy2 = yend + fs / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fs, num_y)

                # one-dimensional FFT in the other direction
                fx1 = xstart + fs / 2
                fx2 = xend + fs / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fs, num_x)

            k_factor = z * dx * dy * self.wavelength

            u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            u0 = u0.squeeze()

            if num_x == 1 and num_y == 1:
                # just 1 number
                return u0.mean()

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_X(xout, self.wavelength)
                #u_out.u = u0.transpose()[: ,0]
                u_out.u = u0[0, :]
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_X(yout, self.wavelength)
                u_out.u = u0.transpose()[:, 0]

                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XY import Scalar_field_XY
                u_out = Scalar_field_XY(xout, yout, self.wavelength)
                u_out.u = u0
                return u_out

        elif num_z > 1:
            u_zs = np.zeros((num_x, num_y, num_z), dtype=complex)
            u_zs = u_zs.squeeze()
            Xout, Yout = np.meshgrid(xout, yout)

            for i, z_now in enumerate(z):
                if verbose is True:
                    print("{}/{}".format(i, num_z), sep="\r", end="\r")
                R = np.sqrt(Xout**2 + Yout**2 + z_now**2)
                F0 = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R) * z_now / R**2 * (1 / R - 1.j * k)

                R = np.sqrt(self.X**2 + self.Y**2 + z_now**2)
                F = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R) * z_now / R**2 * (1 / R - 1.j * k)

                u0 = self.u * F

                # one-dimensional FFT in one direction
                fs = self.wavelength * z_now / dx

                if num_x > 1 and num_y == 1:
                    fx1 = xstart + fs / 2
                    fx2 = xend + fs / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fs, num_x)

                    fy1 = ystart + fs / 2
                    fy2 = yend + fs / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fs, num_y)

                else:
                    fy1 = ystart + fs / 2
                    fy2 = yend + fs / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fs, num_y)

                    fx1 = xstart + fs / 2
                    fx2 = xend + fs / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fs, num_x)

                u0 = F0 * u0

                k_factor = z_now * dx * dy * self.wavelength

                if num_x == 1 and num_y == 1:
                    u_zs[i] = u0.mean() * k_factor
                elif num_x > 1 and num_y == 1:
                    u_zs[:, i] = u0[0, :] * k_factor
                elif num_x == 1 and num_y > 1:
                    u_zs[:, i] = u0.transpose()[:, 0] * k_factor

                elif num_x > 1 and num_y > 1:
                    u_zs[:, :, i] = u0.transpose() * k_factor

            if num_x == 1 and num_y == 1:
                u_out = Scalar_field_Z(z, self.wavelength)
                u_out.u = u_zs
                return u_out

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_XZ(xout, z, self.wavelength)
                u_out.u = u_zs
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_XZ(yout, z, self.wavelength)
                u_out.u = u_zs
                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
                u_out = Scalar_field_XYZ(xout, yout, z, self.wavelength)
                u_out.u = u_zs
                return u_out

        return u_out

    def CZT(self, z, xout=None, yout=None, verbose=False):
        """Chirped Z Transform algorithm for XY Scheme. z, xout, and yout parameters can be numbers or arrays.
        The output Scheme depends on this input parameters.

        Parameters:
            z (float): diffraction distance
            xout (np.array): x array with positions of the output plane
            yout (np.array): y array with positions of the output plane
            verbose (bool): If True, it prints some information

        Returns:
            u_out: Scalar_field_** depending of the input scheme. When all the parameters are numbers, it returns the complex field at that point.


        References:
             [Light: Science and Applications, 9(1), (2020)] 
        """

        if xout is None:
            xout = self.x

        if yout is None:
            yout = self.y

        k = 2 * np.pi / self.wavelength

        if isinstance(z, (float, int)):
            num_z = 1
            # print("z = 0 dim")
        else:
            num_z = len(z)
            # print("z = 1 dim")

        if isinstance(xout, (float, int)):
            num_x = 1
            # print("x = 0 dim")
            xstart = xout
            xend = xout
        else:
            num_x = len(xout)
            # print("x = 1 dim")

            xstart = xout[0]
            xend = xout[-1]

        if isinstance(yout, (float, int)):
            num_y = 1
            # print("y = 0 dim")
            ystart = yout
            yend = yout
        else:
            num_y = len(yout)
            # print("y = 1 dim")

            ystart = yout[0]
            yend = yout[-1]

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        delta_out = np.zeros(2)
        if num_x > 1:
            delta_out[0] = (xend - xstart) / (num_x - 1)

        if num_y > 1:
            delta_out[1] = (yend - ystart) / (num_y - 1)

        Xout, Yout = np.meshgrid(xout, yout)

        if verbose:
            print("num x, num y, num z = {}, {}, {}".format(
                num_x, num_y, num_z))

        if num_z == 1:
            # calculating scalar diffraction below
            # F0 = np.exp(1j * k * z) / (1j * self.wavelength * z) * np.exp(
            #     1j * k / 2 / z * (Xout**2 + Yout**2))
            # F = np.exp(1j * k / 2 / z * (self.X**2 + self.Y**2))

            R1 = np.sqrt(Xout**2 + Yout**2 + z**2)
            R2 = np.sqrt(self.X**2 + self.Y**2 + z**2)

            if z > 0:
                F0 = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R1) * z / R1**2 * (1 / R1 - 1.j * k)
                F = 1 / (2 * np.pi) * np.exp(
                    1.j * k * R2) * z / R2**2 * (1 / R2 - 1.j * k)
            else:
                F0 = 1 / (2 * np.pi) * np.exp(
                    -1.j * k * R1) * z / R1**2 * (1 / R1 + 1.j * k)
                F = 1 / (2 * np.pi) * np.exp(
                    -1.j * k * R2) * z / R2**2 * (1 / R2 + 1.j * k)

            k_factor = z * dx * dy * self.wavelength

            # using Bluestein method to calculate the complex amplitude of the outgoing light beam

            # one-dimensional FFT in one direction
            fsx = self.wavelength * z / dx  # dimension of the imaging plane
            fsy = self.wavelength * z / dy  # dimension of the imaging plane

            if num_x == 1 and num_y == 1:
                u0 = self.u * F
                fy1 = ystart + fsy / 2
                fy2 = yend + fsy / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)
                # one-dimensional FFT in the other direction
                fx1 = xstart + fsx / 2
                fx2 = xend + fsx / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x > 1 and num_y > 1:
                u0 = self.u * F

                fy1 = ystart + fsy / 2
                fy2 = yend + fsy / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                fx1 = xstart + fsx / 2
                fx2 = xend + fsx / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x > 1 and num_y == 1:
                u0 = self.u * F

                # one-dimensional FFT in the other direction
                fx1 = xstart + fsx / 2
                fx2 = xend + fsx / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)

                fy1 = ystart + fsy / 2
                fy2 = yend + fsy / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            elif num_x == 1 and num_y > 1:
                u0 = self.u * F
                fx1 = xstart + fsx / 2
                fx2 = xend + fsx / 2
                u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                fy1 = ystart + fsy / 2
                fy2 = yend + fsy / 2
                u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                # one-dimensional FFT in the other direction
                u0 = F0 * u0 * k_factor  # obtain the complex amplitude of the outgoing light beam

            u0 = u0.squeeze()

            if num_x == 1 and num_y == 1:
                # just 1 number
                return 1j*u0.mean()

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_X(xout, self.wavelength)
                u_out.u = 1j*u0.transpose()[:, 0]
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_X(yout, self.wavelength)
                u_out.u = 1j*u0[:, 0]

                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XY import Scalar_field_XY
                u_out = Scalar_field_XY(xout, yout, self.wavelength)
                u_out.u = 1j*u0
                return u_out

        elif num_z > 1:
            u_zs = np.zeros((num_x, num_y, num_z), dtype=complex)
            u_zs = u_zs.squeeze()
            Xout, Yout = np.meshgrid(xout, yout)

            for i, z_now in enumerate(z):
                if verbose is True:
                    print("{}/{}".format(i, num_z), sep="\r", end="\r")

                R1 = np.sqrt(Xout**2 + Yout**2 + z_now**2)
                R2 = np.sqrt(self.X**2 + self.Y**2 + z_now**2)

                if z_now > 0:
                    F0 = 1 / (2 * np.pi) * np.exp(
                        1.j * k * R1) * z_now / R1**2 * (1 / R1 - 1.j * k)
                    F = 1 / (2 * np.pi) * np.exp(
                        1.j * k * R2) * z_now / R2**2 * (1 / R2 - 1.j * k)
                else:
                    F0 = 1 / (2 * np.pi) * np.exp(
                        -1.j * k * R1) * z_now / R1**2 * (1 / R1 + 1.j * k)
                    F = 1 / (2 * np.pi) * np.exp(
                        -1.j * k * R2) * z_now / R2**2 * (1 / R2 + 1.j * k)

                u0 = self.u * F

                # one-dimensional FFT in one direction
                fsx = self.wavelength * z_now / dx
                fsy = self.wavelength * z_now / dy

                if num_x > 1 and num_y == 1:
                    fx1 = xstart + fsx / 2
                    fx2 = xend + fsx / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)

                    fy1 = ystart + fsy / 2
                    fy2 = yend + fsy / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                elif num_x == 1 and num_y > 1:
                    fy1 = ystart + fsy / 2
                    fy2 = yend + fsy / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                    fx1 = xstart + fsx / 2
                    fx2 = xend + fsx / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                    u0 = u0.transpose()

                elif num_x == 1 and num_y == 1:
                    fy1 = ystart + fsy / 2
                    fy2 = yend + fsy / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                    fx1 = xstart + fsx / 2
                    fx2 = xend + fsx / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)

                elif num_x > 1 and num_y > 1:
                    fy1 = ystart + fsy / 2
                    fy2 = yend + fsy / 2
                    u0 = Bluestein_dft_xy(u0, fy1, fy2, fsy, num_y)

                    fx1 = xstart + fsx / 2
                    fx2 = xend + fsx / 2
                    u0 = Bluestein_dft_xy(u0, fx1, fx2, fsx, num_x)
                u0 = F0 * u0

                k_factor = z_now * dx * dy * self.wavelength

                if num_x == 1 and num_y == 1:
                    u_zs[i] = u0.mean() * k_factor
                elif num_x > 1 and num_y == 1:
                    u_zs[:, i] = u0[0, :] * k_factor
                elif num_x == 1 and num_y > 1:
                    u_zs[:, i] = u0[0, :] * k_factor

                elif num_x > 1 and num_y > 1:
                    u_zs[:, :, i] = u0.transpose() * k_factor

            if num_x == 1 and num_y == 1:
                u_out = Scalar_field_Z(z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

            elif num_x > 1 and num_y == 1:
                u_out = Scalar_field_XZ(xout, z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

            elif num_x == 1 and num_y > 1:
                u_out = Scalar_field_XZ(yout, z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

            elif num_x > 1 and num_y > 1:
                from diffractio.scalar_fields_XYZ import Scalar_field_XYZ
                u_out = Scalar_field_XYZ(xout, yout, z, self.wavelength)
                u_out.u = 1j*u_zs
                return u_out

        return 1j*u_out

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
        z_profile[-1] = z_profile[-2]

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
            plt.colorbar(orientation='vertical', shrink=0.66)

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
        average_intensity = (np.abs(self.u)**2).mean()
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
            matrix (bool): if True numpy.matrix is returned
            new_field (bool): if True it returns a new Scalar_field_XY

        Returns:
            if New_field is True: Scalar_field_X
            if matrix is True: numpy.array
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
            sign (bool): If True, sign is kept, else, it is removed
            matrix (bool): if True numpy.matrix is returned
            new_field (bool): if True it returns a new Scalar_field_XY

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
                 bin_level=None,
                 level0=None,
                 level1=None,
                 new_field=False,
                 matrix=False):
        """Changes the number of points in field, mantaining the area.

        Parameters:
            kind (str): 'amplitude' or 'phase'
            bin_level (float): value of cut. If None, the cut is in the mean value
            level0 (float): minimum value. If None, minimum value of field
            level1 (float): maximum value. If None, maximum value of field
            new_field (bool): if True returns new field
            matrix (bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO: Check and pass to utils
        """

        amplitude = self.get_amplitude(matrix=True, new_field=False)
        phase = self.get_phase(matrix=True, new_field=False)

        if kind == 'amplitude':
            amplitude_binarized = amplitude
            maximum = amplitude.max()
            minimum = amplitude.min()
            if bin_level is None:
                bin_level = (maximum + minimum) / 2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            amplitude_binarized[amplitude <= bin_level] = level0
            amplitude_binarized[amplitude > bin_level] = level1
            u_binarized = amplitude_binarized * phase

        if kind == 'phase':
            phase_binarized = phase
            maximum = phase.max()
            minimum = phase.min()
            if bin_level is None:
                bin_level = (maximum + minimum) / 2
            if level0 is None:
                level0 = minimum
            if level1 is None:
                level1 = maximum

            phase_binarized[phase <= bin_level] = level0
            phase_binarized[phase > bin_level] = level1
            u_binarized = amplitude * phase_binarized

        if new_field is False and matrix is False:
            self.u = u_binarized
            return self.u

        if new_field is False and matrix is True:
            return u_binarized

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = u_binarized
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
            kind (str): "amplitude" o "phase"
            num_levels (int): number of levels for the discretization
            factor (float): from the level, how area is binarized. if 1 everything is binarized,
            phaseInicial (float): *
            new_field (bool): if True returns new field
            matrix (bool): if True it returs a matrix

        Returns:
            Scalar_field_XY: if new_field is True returns Scalar_field_XY

        TODO: Check and pass to utils
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

            u_binarized = discretized_image * phase

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

            u_binarized = amplitude * discretized_image

        if new_field is False and matrix is False:
            self.u = u_binarized
            return

        if new_field is True:
            cn = Scalar_field_XY(self.x, self.y, self.wavelength)
            cn.u = u_binarized
            return cn

        if matrix is True:
            return u_binarized

    def normalize(self, new_field=False):
        """Normalizes the field so that intensity.max()=1.

        Parameters:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, new_field)

    def get_RS_minimum_z(self, n=1, quality=1, verbose=True):
        """Determines the minimum available distance for RS algorithm. If higher or lower quality parameters is required you can add as a parameter

            Args:
                n (float): refraction index of the surrounding medium.
                quality (int, optional): quality. Defaults to 1.
                verbose (bool, optional): prints info. Defaults to True.

            Returns:
                z_min (float): z_min for quality_factor>quality
            """

        range_x = self.x[-1] - self.x[0]
        range_y = self.y[-1] - self.y[0]
        num_x = len(self.x)
        num_y = len(self.y)

        dx = range_x / num_x
        dy = range_y / num_y
        dr_real = np.sqrt(dx**2 + dy**2)
        rmax = np.sqrt(range_x**2 + range_y**2)

        factor = (((quality * dr_real + rmax)**2 -
                   (self.wavelength / n)**2 - rmax**2) / 2 * n /
                  self.wavelength)**2 - rmax**2

        if factor > 0:
            z_min = np.sqrt(factor)
            z_min = z_min / 2
        else:
            z_min = 0

        if verbose:
            if z_min > 1000:
                print("z min = {:2.2f} mm".format(z_min / mm))
            else:
                print("z min = {:2.2f} um".format(z_min))

        return z_min

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
             percentage_intensity=None,
             **kwargs):
        """Draws  XY field.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'contour'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            filename (str): if not '' stores drawing in file,
            cut_value (float): if provided, maximum value to show
            has_colorbar (bool): if True draws the colorbar
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
            reduce_matrix (str): 'standard'
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
                title, colormap_kind, percentage_intensity, **kwargs)
        elif kind == 'field':
            id_fig = self.__draw_field__(logarithm, normalize, title,
                                         cut_value, colormap_kind,
                                         percentage_intensity, **kwargs)
            IDax = None
            IDimage = None
        elif kind == 'real_field':
            id_fig, IDax, IDimage = self.__draw_real_field__(
                logarithm, normalize, title, cut_value, **kwargs)
        else:
            print("not in kinds")

        if has_colorbar in ('horizontal', 'vertical'):
            plt.colorbar(orientation=has_colorbar, shrink=0.5)

        plt.tight_layout()
        if not filename == '':
            plt.savefig(filename,
                        dpi=100,
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
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            cut_value (float): if provided, maximum value to show
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_intensity"]
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)
        id_fig, IDax, IDimage = draw2D(intensity,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)
        plt.tight_layout()

        if self.type == 'Scalar_mask_XY':
            plt.clim(0, 1)

        return id_fig, IDax, IDimage

    def __drawAmplitude__(self,
                          logarithm=False,
                          normalize='maximum',
                          title='intensity',
                          cut_value=1,
                          colormap_kind='',
                          **kwargs):
        """Draws amplitude  XY field.

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            cut_value (float): if provided, maximum value to show
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        amplitude = normalize_draw(amplitude, logarithm, normalize, cut_value)
        max_amplitude = np.abs(amplitude).max()
        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_amplitude"]
        id_fig, IDax, IDimage = draw2D(amplitude,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)
        plt.clim(-max_amplitude, max_amplitude)

        return id_fig, IDax, IDimage

    def __draw_phase__(self,
                       title=r'phase/pi',
                       colormap_kind='',
                       percentage_intensity='None',
                       **kwargs):
        """Draws phase of  XY field

        Parameters:
            title (str): title for the drawing
        """
        amplitude, intensity, phase = field_parameters(self.u,
                                                       has_amplitude_sign=True)
        phase[phase == 1] = -1
        phase = phase / degrees

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        phase[intensity < percentage_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_phase"]

        id_fig, IDax, IDimage = draw2D(phase,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)  # seismic gist_heat
        plt.clim(vmin=-180, vmax=180)

        return id_fig, IDax, IDimage

    def __draw_field__(self,
                       logarithm=False,
                       normalize='maximum',
                       title="",
                       cut_value=None,
                       colormap_kind='',
                       percentage_intensity=None,
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

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        phase[intensity < percentage_intensity * (intensity.max())] = 0

        xsize, ysize = rcParams['figure.figsize']

        #plt.figure(figsize=(2 * xsize, 2 * ysize))
        plt.figure()
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
        plt.colorbar(orientation='horizontal', shrink=0.66, pad=0.1)

        plt.axis('scaled')
        plt.axis(extension)
        plt.title("$intensity$")
        h1.set_cmap(self.CONF_DRAWING["color_intensity"])
        if self.type == 'Scalar_mask_XY':
            plt.clim(0, 1)

        plt.subplot(1, 2, 2)
        phase = phase / degrees

        # elimino la fase en la visualicion cuando no hay campo
        h2 = plt.imshow(phase,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel("$x  (\mu m)$")
        plt.ylabel("$y  (\mu m)$")
        plt.colorbar(orientation='horizontal', shrink=0.66, pad=0.1)

        plt.axis('scaled')
        plt.axis(extension)
        plt.title("$phase$")
        plt.clim(-180, 180)
        h2.set_cmap(self.CONF_DRAWING["color_phase"])  #
        plt.subplots_adjust(0.01, 0.01, 0.99, 0.95, 0.35, 0.35)
        return (h1, h2)

    def __draw_real_field__(self,
                            logarithm=False,
                            normalize='maximum',
                            cut_value=1,
                            title="",
                            colormap_kind='',
                            percentage_intensity=None,
                            **kwargs):
        """Draws real field  XY field.

        Parameters:
            logarithm(bool): If True, intensity is scaled in logarithm
            normalize(str):  False, 'maximum', 'area', 'intensity'
            title(str): title for the drawing
            cut_value(float): if provided, maximum value to show
        """
        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        rf = np.real(self.u)
        intensity = np.abs(self.u)**2
        rf[intensity < percentage_intensity * (intensity.max())] = 0

        if colormap_kind in ['', None, []]:
            colormap_kind = self.CONF_DRAWING["color_real"]

        id_fig, IDax, IDimage = draw2D(rf,
                                       self.x,
                                       self.y,
                                       xlabel="$x  (\mu m)$",
                                       ylabel="$y  (\mu m)$",
                                       title=title,
                                       color=colormap_kind,
                                       reduce_matrix=self.reduce_matrix,
                                       **kwargs)

        return id_fig, IDax, IDimage

    def video(self,
              kind,
              zs,
              logarithm=False,
              normalize=False,
              time_video=10 * seconds,
              frames_reduction=1,
              filename='video.avi',
              dpi=100):
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
        return 1 / (2 * pi) * exp(-1.j * k * R) * z / R**2 * (1 / R + 1.j * k)
    elif kind == 'x':
        return 1 / (2 * pi) * exp(-1.j * k * R) * X / R**2 * (1 / R + 1.j * k)
    elif kind == 'y':
        return 1 / (2 * pi) * exp(-1.j * k * R) * Y / R**2 * (1 / R + 1.j * k)
    elif kind == '0':
        return 1 / (2 * pi) * exp(-1.j * k * R) / R * (1 / R + 1.j * k)


def kernelFresnel(X, Y, wavelength=0.6328 * um, z=10 * mm, n=1):
    """
    Kernel for Fesnel propagation.

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


def get_RS_minimum_z(range_x,
                     range_y,
                     num_x,
                     num_y,
                     wavelength,
                     n=1,
                     quality=1,
                     verbose=True):
    """_summary_

    Args:
        range_x (float): range_x
        range_y (float): range_y
        num_x (int): num_x
        num_y (int): num_y
        z (float): z
        wavelength (float): wavelength
        quality (int, optional): quality. Defaults to 1.
        verbose (bool, optional): prints info. Defaults to True.

    Returns:
        z_min (float): z_min for quality_factor>quality
    """

    dx = range_x / num_x
    dy = range_y / num_y
    dr_real = np.sqrt(dx**2 + dy**2)
    rmax = np.sqrt(range_x**2 + range_y**2)

    factor = (
        ((quality * dr_real + rmax)**2 -
         (wavelength / n)**2 - rmax**2) / 2 * n / wavelength)**2 - rmax**2

    if factor > 0:
        zmin = np.sqrt(factor)
        zmin = zmin / 2
    else:
        zmin = 0

    if verbose:
        print("z min = {:2.2f}".format(zmin))

    return zmin


def quality_factor(range_x,
                   range_y,
                   num_x,
                   num_y,
                   z,
                   wavelength,
                   n=1,
                   verbose=False):
    """Determine the quality factor for RS algorithm

    Args:
        x (np.array): x array with positions
        y (np.array): y array with positions
        z (float): observation distance
        wavelength (float): wavelength)
        n (float): refraction index

    Returns:
        _type_: _description_
    """

    dx = range_x / num_x
    dy = range_y / num_y
    dr_real = np.sqrt(dx**2 + dy**2)
    rmax = np.sqrt(range_x**2 + range_y**2)

    dr_ideal = np.sqrt((wavelength / n)**2 + rmax**2 + 2 *
                       (wavelength / n) * np.sqrt(rmax**2 + z**2)) - rmax
    quality = dr_ideal / dr_real
    quality = quality * 2

    if verbose:
        print("Quality factor = {:2.2f}".format(quality))

    return quality

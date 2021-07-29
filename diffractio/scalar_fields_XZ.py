# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_field_XZ class.

It includes multiprocessing for RS and BPM polychromatic

It can be considered an extension of Scalar_field_X for visualizing XZ fields

For the case of Rayleigh sommefeld it is not necessary to compute all z positions but the final.

Nevertheless, for BPM method, intermediate computations are required. In this class, intermediate results are stored.

X,Z fields are defined using ndgrid (not with meshgrid, it is different).

It is required also for generating masks and fields.
The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refraction index XZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for XZ scalar fields*

*Definition of a scalar field*
    * instatiation, clean_refraction_index
    * save, load data
    * rotate_field, cut_resample,

*Illumination*
    * incident_field

*Operations*
    * surface_detection
    * search focus

*Propagation*
    * RS, RS_polychormatic,
    * BPM, BPM_poychromatic, BPM_inverse, BPM_back_propagation

*Drawing functions*
    * draw
    * draw_refraction_index
    * draw_incident_field
    * video_profiles
    * draw_profiles_interactive

*Parameters*
    * final_field
    * profile_longitudinal
    * profile_transversal
"""

import copyreg
import sys
import time
import types
from copy import deepcopy
from multiprocessing import Pool

import matplotlib.animation as animation
import matplotlib.cm as cm
from numpy import array, concatenate, diff, gradient, pi, sqrt, zeros
from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fft2, fftshift, ifft, ifft2
from scipy.interpolate import RectBivariateSpline

from . import (degrees, eps, mm, np, num_max_processors, params_drawing, plt,
               seconds, um)
from .scalar_fields_X import (PWD_kernel, Scalar_field_X, WPM_schmidt_kernel,
                              kernelRS, kernelRSinverse)
from .scalar_masks_X import Scalar_mask_X
from .scalar_sources_X import Scalar_source_X
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw, prepare_drawing, prepare_video
from .utils_math import get_k, ndgrid, nearest, rotate_image
from .utils_multiprocessing import _pickle_method, _unpickle_method
from .utils_optics import FWHM1D, beam_width_1D, field_parameters

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Scalar_field_XZ(object):
    """Class for working with XZ scalar fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        z (numpy.array): linear array wit equidistant positions for z values
        wavelength (float): wavelength of the incident field
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        self.z (numpy.array): linear array wit equidistant positions for z values
        self.wavelength (float): wavelength of the incident field.
        self.u0 (numpy.array): (x) size x - field at the last z position
        self.u (numpy.array): (x,z) complex field
        self.n_background (numpy.array): (x,z) refraction index
        self.fast (bool): if True fast algoritm (approx. Hankle function)
        self.info (str): String with info about the simulation
    """

    def __init__(self,
                 x=None,
                 z=None,
                 wavelength=None,
                 n_background=1,
                 info=''):

        self.x = x
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        self.fast = False
        self.quality = 0
        self.borders = None  # borders at refraction index
        self.params_drawing = params_drawing

        if x is not None and z is not None:
            self.X, self.Z = ndgrid(x, z)
            self.u0 = Scalar_field_X(x, wavelength)
            self.u = np.zeros_like(self.X, dtype=complex)
            self.n = n_background * np.ones(np.shape(
                self.X), dtype=complex)  # el indice de refracción
        else:
            self.X = None
            self.Z = None
            self.u0 = None
            self.u = None
            self.n = None
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XZ'
        self.date = get_date()

    def __str__(self):
        """Represents main data of the atributes"""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - x:  {},   z:  {},   u:  {}".format(
            self.type, self.x.shape, self.z.shape, self.u.shape))
        print(" - xmin:       {:2.2f} um,  xmax:      {:2.2f} um".format(
            self.x[0], self.x[-1]))
        print(" - zmin:       {:2.2f} um,  zmax:      {:2.2f} um".format(
            self.z[0], self.z[-1]))
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
            kind (str): instruction how to add the fields: - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1. - 'standard': add fields u3=u1+u2 and does nothing.

        Returns:
            Scalar_field_X: `u3 = u1 + u2`
        """

        u3 = Scalar_field_XZ(self.x, self.z, self.wavelength,
                             self.n_background)
        u3.n = self.n

        if kind == 'standard':
            u3.u = self.u + other.u

        elif kind == 'maximum1':
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            f1 = np.angle(self.u)
            f2 = np.angle(other.u)
            t3 = t1 + t2
            t3[t3 > 0] = 1.
            u3.u = t3 * np.exp(1j * (f1 + f2))

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

        u3 = Scalar_field_XZ(self.x, self.z, self.wavelength,
                             self.n_background)
        u3.n = self.n
        u3.u = self.u - other.u
        return u3

    def __rotate__(self, angle, position=None):
        """Rotation of X,Z with respect to position

        Parameters:
            angle (float): angle to rotate, in radians
            position (float, float): position of center of rotation
        """

        if position is None:
            x0 = (self.x[-1] + self.x[0]) / 2
            z0 = (self.z[-1] + self.z[0]) / 2
        else:
            # Definicion de la rotation
            x0, z0 = position

        Xrot = x0 + (self.X - x0) * np.cos(angle) + (self.Z -
                                                     z0) * np.sin(angle)
        Zrot = z0 - (self.X - x0) * np.sin(angle) + (self.Z -
                                                     z0) * np.cos(angle)
        return Xrot, Zrot

    def rotate_field(self, angle, center_rotation, kind='all', n_background=1):
        """Rotate all the image a certain angle

        Parameters:
            angle (float): angle to rotate, in radians
            n_background (float): refraction index of zone incoming
            kind (str): 'all', 'n', 'field'
            center_rotation (float, float): (z,x) position for rotation
        """
        angle = -angle

        if kind in ('n', 'all'):
            n_real_rotate = rotate_image(self.z, self.x, np.real(self.n),
                                         angle * 180 / pi, center_rotation)
            n_imag_rotate = rotate_image(self.z, self.x, np.imag(self.n),
                                         angle * 180 / pi, center_rotation)
            n_rotate = n_real_rotate + 1j * n_imag_rotate
            n_rotate[n_rotate < n_background] = n_background
            self.n = n_rotate
            self.n[self.n < 1.2] = self.n_background

        self.surface_detection(mode=1, min_incr=0.1, reduce_matrix='standard')

        if kind in ('field', 'all'):
            u_real_rotate = rotate_image(self.z, self.x, np.real(self.u),
                                         angle * 180 / pi, center_rotation)
            u_imag_rotate = rotate_image(self.z, self.x, np.imag(self.u),
                                         angle * 180 / pi, center_rotation)
            u_rotate = u_real_rotate + 1j * u_imag_rotate
            self.u = u_rotate

        if kind == 'n':
            self.u = np.zeros_like(self.u)

    def clear_field(self):
        """clear field"""
        self.u = np.zeros(np.shape(self.u), dtype=complex)

    def clear_refraction_index(self):
        """clear refraction index n(x,z)=n_background"""

        self.n = self.n_background * np.ones_like(self.X, dtype=complex)

    def mask_field(self, size_edge=0):
        """
        mask the incident field at the edges, each edge is masked size_edge

        Parameters:
            size_edge (float): size of edges
        """

        L = self.x[-1] - self.x[0]
        x_center = (self.x[-1] + self.x[0]) / 2
        mask = Scalar_mask_X(x=self.x, wavelength=self.wavelength)
        mask.slit(x0=x_center, size=L - 2 * size_edge)
        self.u0.u = self.u0.u * mask.u

    def filter_refraction_index(self,
                                type_filter=2,
                                pixels_filtering=10,
                                max_diff_filter=0.1,
                                draw_check=False):
        """
        Technique to remove artifacts in BPM propagation.

        Parameters:
            type_filter (int): 1 - 2D, 2 - 1D z (selective), 3 - 1D x (selective)
            pixels_filtering (int): num_pixels used for filtering
            max_diff_filter (float): maximum difference of n in profile between two adjancted pixels to use selective filtering
            draw_check (bool): draw the differences.

        Returns:
            (float): percentaje_filtered
            (np.array): lineas_filtradas

        References:
            Robert McLeod "Numerical Methods in Photonics Lecture Notes"  University of Colorado at Boulder, pag 204 (15/54)
        """

        if draw_check is True:
            indice_sin_variar = deepcopy(self.n)

        num_filtrados = 0
        if type_filter == 1:
            # Filtro 2D, pero solo ejecuta en una dirección
            lineas_filtradas = np.ones_like(self.z)

            filtro1 = np.zeros_like(self.n)
            sizex, sizez = self.n.shape
            centerx, centerz = int(sizex / 2), int(sizez / 2)
            filtro1[centerx - pixels_filtering:centerx + pixels_filtering,
                    centerz - 1:centerz + 1] = 1
            filtro1 = filtro1 / sum(sum(filtro1))
            self.n = fftshift(ifft2(fft2(self.n) * fft2(filtro1)))
        elif type_filter == 2:
            # Filtro 1D, solo ejecuta cuando hay diferencias de índice eje x
            lineas_filtradas = np.zeros_like(self.z)
            filtro1 = np.zeros_like(self.x)
            sizex = len(filtro1)
            centerx = (self.x[-1] + self.x[0]) / 2
            # i_centerx = int(sizex / 2)
            # filtro1[i_centerx - pixels_filtering:i_centerx + pixels_filtering] = 1
            filtro1 = np.exp(-(self.x - centerx)**2 /
                             (2 * pixels_filtering**2))
            filtro1 = filtro1 / sum(filtro1)
            for i in range(len(self.z)):
                max_diff = np.abs(np.diff(self.n[:, i])).max()
                if max_diff > max_diff_filter:
                    lineas_filtradas[i] = 1
                    self.n[:, i] = fftshift(
                        ifft(fft(self.n[:, i]) * fft(filtro1)))
                    num_filtrados = num_filtrados + 1
            percentaje_filtered = 100 * num_filtrados / len(self.z)
        elif type_filter == 3:
            # Filtro 1D, solo ejecuta cuando hay diferencias de índice eje z
            lineas_filtradas = np.zeros_like(self.x)
            filtro1 = np.zeros_like(self.z)
            sizez = len(filtro1)
            centerz = int(sizez / 2)
            filtro1[centerz - pixels_filtering:centerz + pixels_filtering] = 1
            filtro1 = filtro1 / sum(filtro1)
            for i in range(len(self.x)):
                max_diff = np.abs(np.diff(self.n[i, :])).max()
                if max_diff > max_diff_filter:
                    lineas_filtradas[i] = 1
                    self.n[i, :] = fftshift(
                        ifft(fft(self.n[i, :]) * fft(filtro1)))
                    num_filtrados = num_filtrados + 1
            percentaje_filtered = 100 * num_filtrados / len(self.x)

        if draw_check is True:
            plt.figure()
            plt.plot(self.z, lineas_filtradas)
            plt.xlabel('z ($\mu m$)')
            plt.ylabel('filtered zone)')
            plt.title("detection of edges", fontsize=24)

            diferencias_indice = np.abs(self.n - indice_sin_variar)

            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

            plt.figure()

            h1 = plt.imshow(diferencias_indice,
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
            plt.xlabel('z ($\mu m$)')
            plt.ylabel('x ($\mu m$)')

            plt.axis(extension)
            h1.set_cmap(cm.gray_r)
            plt.title("filter_refraction_index: n variations", fontsize=24)

        return percentaje_filtered, lineas_filtradas

    def discretize_refraction_index(self, n_layers):
        """Takes a refraction index an discretize it according refraction indexes.

        Parameters:
            n_layers (np.array): array with refraction indexes to discretize
        """
        n = deepcopy(self.n)
        for i in range(len(n_layers) - 1):
            i_capa = np.bitwise_and(self.n >= n_layers[i],
                                    self.n <= n_layers[i + 1])
            n_central = (n_layers[i] + n_layers[i + 1]) / 2
            n[i_capa] = n_central
        self.n = n
        return n

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
        """Load data from a file to a Scalar_field_XZ.
            The methods included are: npz, matlab


        Parameters:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename, verbose)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

    def cut_resample(self,
                     x_limits='',
                     z_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind=(3, 1)):
        """it cut the field to the range (x0,x1). if one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Parameters:
            x_limits (float,float): (x0,x1) starting and final points to cut. if '' - takes the current limit x[0] and x[-1]
            z_limits (float,float): (z0,z1) - starting and final points to cut. if '' - takes the current limit z[0] and z[-1]
            num_points (int): it resamples x, z and u. ([],'',0,None) -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XZ
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if z_limits == '':
            # used only for resampling
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)
        # new_num_points = i_x1 - i_x0
        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_z = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            z_new = np.linspace(z0, z1, num_points_z)
            X_new, Z_new = np.meshgrid(x_new, z_new)

            f_interp_abs = RectBivariateSpline(self.x,
                                               self.z,
                                               np.abs(self.u),
                                               kx=kxu,
                                               ky=kxu,
                                               s=0)
            f_interp_phase = RectBivariateSpline(self.x,
                                                 self.z,
                                                 np.angle(self.u),
                                                 kx=kxu,
                                                 ky=kxu,
                                                 s=0)
            u_new_abs = f_interp_abs(x_new, z_new)
            u_new_phase = f_interp_phase(x_new, z_new)
            u_new = u_new_abs * np.exp(1j * u_new_phase)

            n_interp_real = RectBivariateSpline(self.x,
                                                self.z,
                                                np.real(self.n),
                                                kx=kxn,
                                                ky=kxn,
                                                s=0)
            n_interp_imag = RectBivariateSpline(self.x,
                                                self.z,
                                                np.imag(self.n),
                                                kx=kxn,
                                                ky=kxn,
                                                s=0)
            n_new_real = n_interp_real(x_new, z_new)
            n_new_imag = n_interp_imag(x_new, z_new)
            n_new = n_new_real + 1j * n_new_imag

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_z0, i_z1)
            x_new = self.x[i_s]
            z_new = self.z[j_s]
            X_new, Z_new = np.meshgrid(x_new, z_new)
            u_new = self.u[i_s, j_s]
            n_new = self.n[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.z = z_new
            self.u = u_new
            self.n = n_new
            self.X = X_new
            self.Z = Z_new
        elif new_field is True:
            field = Scalar_field_XZ(x=x_new,
                                    z=z_new,
                                    wavelength=self.wavelength)
            field.u = u_new
            field.n = n_new
            return field

    def incident_field(self, u0, z0=None):
        """Incident field for the experiment. It takes a Scalar_source_X field

        Parameters:
            u0 (Scalar_source_X): field produced by Scalar_source_X (or a X field)
            z0 (float): position of the incident field. if None, '', [], is at the beginning
        """

        if z0 in (None, '', []):
            self.u0 = u0
        else:
            iz, _, _ = nearest(self.z, z0)
            self.u[:, iz] = self.u[:, iz] + u0.u

    def final_field(self):
        """Returns the final field as a Scalar_field_X."""

        u_final = Scalar_field_X(x=self.x,
                                 wavelength=self.wavelength,
                                 n_background=self.n_background,
                                 info="from final_field at z0= {} um".format(
                                     self.z[-1]))
        u_final.u = self.u[:, -1]
        return u_final

    def __BPM__(self, matrix=False, verbose=False):
        """Beam propagation method (BPM).

        Parameters:
            matrix (bool): if True returns matrix, else goes to self.u
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119

        """
        dn = np.abs(np.diff(self.n).max())
        dz = self.z[1] - self.z[0]

        q1 = (0.25 * self.wavelength / 2 * dn / dz,
              0.25 * (self.x[-1] - self.x[0])**2 / self.wavelength / dz)
        self.quality = q1

        k0 = 2 * np.pi / self.wavelength

        numz = len(self.z)  # distance en z
        numx = len(self.x)  # distance en x
        deltaz = self.z[1] - self.z[0]  # Tamaño del sampling
        rangox = self.x[-1] - self.x[0]

        pixelx = np.linspace(-int(numx / 2), int(numx / 2), numx)
        # initial field
        field_z = self.u0.u
        # Calculo de la phase 1 normalizada -------------------
        kx1 = np.linspace(0, int(numx / 2) + 1, int(numx / 2))
        kx2 = np.linspace(-int(numx / 2), -1, int(numx / 2))
        # Número de ondas del material en una dimensión
        kx = (2 * np.pi / rangox) * np.concatenate((kx1, kx2))
        # Función de transferencia para la propagación que es identica
        # a la respuesta de frecuencia espacial en óptica de Fourier
        # incorporando el termino (-j k0 z).
        phase1 = np.exp((-1j * deltaz * kx**2) / (2 * k0))
        # Campo en el índice de refracción
        field = np.zeros(np.shape(self.n), dtype=complex)
        # Función supergausiana para eliminar rebotes en los edges
        filter_edge = np.exp(-((pixelx) / (0.99 * 0.5 * numx))**90)  # 0.98

        field[:, 0] = field_z
        for k in range(0, numz):
            if verbose is True:
                if sys.version_info.major == 3:
                    print("BPM: {}/{}".format(k, numz), sep="\r", end="\r")
                else:
                    print("BPM: {}/{}".format(k,
                                              numz).format(k, numz))

            phase2 = np.exp(1j * self.n[:, k] * k0 * deltaz)
            field_z = ifft((fft(field_z) * phase1)) * phase2
            field_z = field_z * filter_edge + self.u[:, k]
            field[:, k] = field_z

        if matrix is True:
            return field
        else:
            self.u = field

    def BPM(self, division=False, matrix=False, verbose=False):
        """Beam propagation method (BPM).

        Parameters:
            division (False, int): If False nothing, else divides the BPM algorithm in several different executions. To avoid RAM problems
            matrix (bool): if True returns a matrix else
            verbose (bool): shows data process by screen

        References:
           Algorithm in "Engineering optics with matlab" pag 119.
        """

        t1 = time.time()

        if division is False:
            # standard BPM _algorithm
            self.__BPM__(matrix, verbose)

        else:
            # Here is the division of self.z in parts
            num_executions = int(np.ceil(len(self.z) / division))
            uf = self.u0
            for i in range(num_executions):
                if verbose is True:
                    if sys.version_info.major == 3:
                        print("{}/{}".format(i, num_executions),
                              sep="\r",
                              end="\r")
                    else:
                        print("{}/{}".format(i, num_executions))
                sl = slice(i * division, (i + 1) * division)
                ui = Scalar_field_XZ(x=self.x,
                                     z=self.z[sl],
                                     wavelength=self.wavelength,
                                     n_background=self.n_background)
                ui.n = self.n[:, sl]
                ui.u0 = uf

                ui.BPM()
                uf = ui.final_field().u
                self.u[:, sl] = ui.u

            if matrix is True:
                return self.u
        if verbose is True:
            t2 = time.time()
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))

    def BPM_inverse(self, verbose=False):
        """
        Beam propagation method (BPM) in inverse mode.

        Parameters:
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        c_inverse = Scalar_field_XZ(x=self.x,
                                    z=self.z,
                                    wavelength=self.wavelength,
                                    n_background=self.n_background)
        c_inverse.n = np.fliplr(self.n)
        c_inverse.u0.u = np.conjugate(self.u[:, -1])
        c_inverse.u = np.zeros_like(self.u)
        c_inverse.BPM(verbose)

        # ATENCIÓN, HAGO LA INVERSA Y LA REPRESENTACIÓN ES IGUAL QUE LA DIRECTA
        c_inverse.n = (np.fliplr(c_inverse.n))
        c_inverse.u = (np.fliplr(c_inverse.u))
        return c_inverse

    def BPM_back_propagation(self, verbose=False):
        """
        Beam propagation method (BPM). The field that generates the final field is obtained.

        Parameters:
            verbose (bool): shows data process by screen

        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        c_backpropagation = Scalar_field_XZ(x=self.x,
                                            z=self.z,
                                            wavelength=self.wavelength,
                                            n_background=self.n_background)
        c_backpropagation.n = np.fliplr(self.n)
        c_backpropagation.u = np.fliplr(self.u)
        c_backpropagation.BPM(verbose)
        c_backpropagation.n = (np.fliplr(c_backpropagation.n))
        c_backpropagation.u = (np.fliplr(c_backpropagation.u))
        return c_backpropagation

    def __RS_multiprocessing__(self, i):
        """Internal for multiprocessing

        TODO:
            It can be using a dict as input
        """
        if self.z.min() > 0:
            H = kernelRS(self.xtemp,
                         self.wavelength,
                         self.z[i],
                         self.n_background,
                         kind='z',
                         fast=self.fast)
        else:
            H = kernelRSinverse(self.xtemp,
                                self.wavelength,
                                self.z[i],
                                self.n_background,
                                kind='z',
                                fast=self.fast)

        dx = self.x[1] - self.x[0]

        S = ifft(fft(self.Utemp) * fft(H)) * dx
        nx = len(self.x)
        return S[nx - 1:]

    def RS(self,
           xout=None,
           yout=None,
           verbose=False,
           num_processors=num_max_processors):
        """Rayleigh Sommerfeld propagation algorithm

        Parameters:
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
           time in the processing
        """

        time1 = time.time()

        xin = self.x
        x1 = self.x[0]
        xin1 = self.x[0]

        if xout is None:
            xout = self.x
        nx = len(xout)
        dx = xout[1] - xout[0]

        # parametro de quality
        dr_real = sqrt(dx**2)
        rmax = sqrt((xout**2).max())
        dr_ideal = sqrt((self.wavelength / self.n_background)**2 + rmax**2 +
                        2 * (self.wavelength / self.n_background) *
                        sqrt(rmax**2 + self.z.min()**2)) - rmax
        self.quality = dr_ideal / dr_real

        # when computation is performed: quality is determined
        if (self.quality < 1):
            print('- Needs denser sampling: factor: {:2.2f}'.format(
                self.quality))
        else:
            if verbose is True:
                if sys.version_info.major == 3:
                    print('Good result: factor {:2.2f}'.format(self.quality),
                          sep="\r",
                          end="\r")
                else:
                    print('Good result: factor {:2.2f}'.format(self.quality))
                print()

        # matrix W para integracion simpson
        a = [2, 4]
        num_rep = int(round((nx) / 2) - 1)
        # print(num_rep)

        b = array(a * num_rep)
        W = concatenate(((1, ), b, (2, 1))) / 3.

        if float(nx) / 2 == round(nx / 2):  # es par
            i_central = num_rep + 1
            W = concatenate((W[:i_central], W[i_central + 1:]))

        xext = x1 - xin[::-1]  # da la vuelta
        xext = xext[0:-1]
        xext = concatenate((xext, self.x - xin1))

        # field
        U = zeros(2 * nx - 1, dtype=complex)
        U[0:nx] = W * self.u0.u
        self.Utemp = U

        self.xtemp = xext

        if num_processors == 1:
            for i_z, z in enumerate(self.z):

                if z >= 0:
                    H = kernelRS(xext,
                                 self.wavelength,
                                 z,
                                 self.n_background,
                                 kind='z',
                                 fast=self.fast)
                else:
                    H = kernelRSinverse(xext,
                                        self.wavelength,
                                        z,
                                        self.n_background,
                                        kind='z',
                                        fast=self.fast)

                # calculo de la transformada de Fourier
                S = ifft(fft(U) * fft(H)) * dx
                self.u[:, i_z] = S[nx - 1:]  # hasta el final
        else:
            pool = Pool(num_processors)
            t = pool.map(self.__RS_multiprocessing__, list(range(len(self.z))))
            pool.close()
            pool.join()
            for i in range(len(self.z)):
                self.u[:, i] = t[i]
        time2 = time.time()

        if verbose is True:
            if sys.version_info.major == 3:
                print("time in RS= {}. num proc= {}".format(
                    time2 - time1, num_processors),
                    sep="\r",
                    end="\r")
            else:
                print("time in RS= {}. num proc= {}".format(
                    time2 - time1, num_processors))

        return self.u

    def PWD(self, n=None, matrix=False, verbose=False):
        """
        Plane wave decomposition algorithm (PWD).

        Parameters:
            n (np. array): refraction index, If None, it is n_background
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints state of algorithm

        Returns:
            numpy.array(): Field at at distance dz from the incident field

        References:
            1. Schmidt, S. et al. Wave-optical modeling beyond the thin-element-approximation. Opt. Express 24, 30188 (2016).

        """
        dx = self.x[1] - self.x[0]
        dz = self.z[1] - self.z[0]
        k0 = 2 * np.pi / self.wavelength

        if n is None:
            n = self.n_background

        kx = get_k(self.x, '+')

        K_perp2 = kx**2
        num_steps = len(self.z)
        self.clear_field()
        self.u[:, 0] = self.u0.u
        for i, zi in enumerate(self.z[0:-1]):
            result = self.u[:, i]
            result_next = PWD_kernel(result, n, k0, K_perp2, dz)
            self.u[:, i + 1] = result_next
            if sys.version_info.major == 3:
                print("{}/{}".format(i, num_steps), sep='\r', end='\r')
            else:
                print("{}/{}".format(i, num_steps))

        if matrix is True:
            return self.u

    def WPM(self,
            kind='schmidt',
            has_edges=True,
            pow_edge=80,
            matrix=False,
            verbose=False):
        """
        WPM Methods.
        'schmidt method is very fast, only needs discrete number of refraction indexes'


        Parameters:
            kind (str): 'schmidt, scalar, TE, TM
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength
        x = self.x
        z = self.z
        dx = x[1] - x[0]
        dz = z[1] - z[0]

        self.u[:, 0] = self.u0.u
        kx = get_k(x, flavour='+')
        k_perp2 = kx**2
        k_perp = kx

        X, KP = np.meshgrid(x, k_perp)

        if has_edges is False:
            filter_edge = 1
        else:
            filter_edge = np.exp(-(self.x / (self.x[0]))**pow_edge)

        t1 = time.time()

        num_steps = len(self.z)
        for j in range(1, num_steps):

            if kind == 'schmidt':
                self.u[:, j] = self.u[:, j] + WPM_schmidt_kernel(
                    self.u[:, j - 1], self.n[:, j - 1], k0, k_perp2,
                    dz) * filter_edge
            else:
                print("{}".format(j), sep='\r', end='\r')

                Nj, KP = np.meshgrid(self.n[:, j - 1], k_perp)
                X, UJ = np.meshgrid(x, fftshift(fft(self.u[:, j - 1])))

                kz = csqrt((Nj * k0)**2 - KP**2)
                Pj = np.exp(1j * kz * dz)

                if kind == 'TM':
                    M00, _, _, M11 = self.M_xz(j=j, kx=kx)
                    M = M00
                elif kind == 'TE':
                    M00, _, _, M11 = self.M_xz(j=j, kx=kx)
                    M = M11
                elif kind == 'scalar':
                    M = 1

                WXj = M * UJ * np.exp(1j * KP * (X - x[0]))

                uj = np.mean(WXj * Pj, 0)
                self.u[:, j] = uj * filter_edge + self.u[:, j]

            if verbose is True:
                if sys.version_info.major == 3:
                    print("{}/{}".format(j, num_steps), sep='\r', end='\r')
                else:
                    print("{}/{}".format(j, num_steps))

        t2 = time.time()
        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))

        if matrix is True:
            return self.u

    def M_xz(self, j, kx):
        """
        Refraction matrix given in eq. 18 from  M. W. Fertig and K.-H. Brenner,
        “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.
        """
        # simple parameters

        k0 = 2 * np.pi / self.wavelength

        z = self.z
        x = self.x

        num_x = self.x.size
        num_z = self.z.size

        dz = z[1] - z[0]
        dx = x[1] - x[0]

        nj = self.n[:, j]
        nj_1 = self.n[:, j - 1]

        n_med = nj.mean()
        n_1_med = nj_1.mean()

        # parameters
        NJ, KX = np.meshgrid(nj, kx)
        NJ_1, KX = np.meshgrid(nj_1, kx)

        k_perp = KX + eps

        kz_j = np.sqrt((n_med * k0)**2 - k_perp**2)
        kz_j_1 = np.sqrt((n_1_med * k0)**2 - k_perp**2)

        tg_TM = 2 * NJ_1**2 * kz_j / (NJ**2 * kz_j_1 + NJ_1**2 * kz_j)
        t_TE = 2 * kz_j_1 / (kz_j_1 + kz_j)

        kj_1 = NJ_1 * k0
        fj_1 = k_perp**2 / (NJ_1**2 * kj_1**2)
        # cuidado no es la misma definición, me parece que está repetido
        e_xj_1 = np.gradient(NJ_1**2, dx, axis=0)
        eg_xj_1 = fj_1 * e_xj_1
        # cuidado no es la misma definición, me parece que está mal por no ser simétrica

        p001 = 0
        p002 = tg_TM * (1 - 1j * eg_xj_1)
        p111 = t_TE
        p112 = 0

        # M00
        M00 = (p001 + p002)

        # M01
        M01 = 0

        # M10
        M10 = 0

        # M11
        M11 = (p111 + p112)

        return M00, M01, M10, M11

    def RS_polychromatic(self,
                         initial_field,
                         wavelengths,
                         spectrum='',
                         verbose=False,
                         num_processors=num_max_processors):
        """Rayleigh Sommerfeld propagation algorithm for polychromatic light.

        Parameters:
            initial_field (Scalar_field_X): function with only input variable wavelength
            wavelengths (numpy.array): array with wavelengths
            spectrum (numpy.array): array with spectrum. if '' then uniform_spectrum
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Scalar_field_XZ: self.u=sqrt(Intensities) - no phase is stored, only intensity
        """
        if isinstance(spectrum, np.ndarray):
            pass
        elif spectrum in ('', None, [], 0):
            spectrum = np.ones_like(wavelengths)

        I_final = np.zeros_like(self.u, dtype=float)
        u_temp = Scalar_field_XZ(self.x, self.z, self.wavelength,
                                 self.n_background)
        for i, wavelength in enumerate(wavelengths):
            if verbose:
                print(i)
            u_temp.wavelength = wavelength
            self.u = np.zeros_like(self.X, dtype=complex)
            u_ini = initial_field(wavelength)
            u_temp.clear_field()
            u_temp.incident_field(u_ini)
            u_temp.RS(verbose=verbose, num_processors=num_processors)
            I_final = I_final + spectrum[i] * np.abs(u_temp.u)**2

        u_temp.u = np.sqrt(I_final)
        return u_temp

    def BPM_polychromatic(self,
                          initial_field,
                          wavelengths,
                          spectrum,
                          verbose=False,
                          num_processors=4):
        """Rayleigh Sommerfeld propagation algorithm for polychromatic light

        Parameters:
            initial_field (Scalar_field_X): function with only input variable wavelength
            wavelengths (numpy.array): array with wavelengths
            spectrum (numpy.array): array with spectrum. if '' then uniform_spectrum
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
            Scalar_field_XZ: self.u=sqrt(Intensities) - no phase is stored, only intensity
        """

        if isinstance(spectrum, np.ndarray):
            pass
        elif spectrum in ('', None, [], 0):
            spectrum = np.ones_like(wavelengths)

        I_final = np.zeros_like(self.u, dtype=float)
        for i, wavelength in enumerate(wavelengths):
            u_temp = initial_field(wavelength)
            u_temp.BPM(verbose=False)
            I_final = I_final + spectrum[i] * np.abs(u_temp.u)**2
        u_temp.u = np.sqrt(I_final)
        return u_temp

    def fast_propagation(self, mask_xz, num_pixels_slice=1024, verbose=False):
        """combines RS and BPM"" to generate the final field

        Parameters:
            mask_xz (Scalar_mask_XZ): function that returns Scalar_mask_XZ
            num_pixels_slice (int): num of slices for each BPM propagation
            verbose (bool): If True prints info.

        Returns:

        """

        # check which parts are constant
        z_transitions, algorithm, refr_index_RS = self._detect_transitions_()

        if verbose:
            print(("z_transitions={}".format(z_transitions)))
            print(("algorithm={}".format(algorithm)))
            print(("refr_index_RS={}".format(refr_index_RS)))
        transitions = (z_transitions, algorithm, refr_index_RS)

        fields_BPM = []
        u_current = Scalar_source_X(x=self.x, wavelength=self.wavelength)
        u_current.u = self.u0.u

        for i_zone in range(len(z_transitions) - 1):
            if algorithm[i_zone] == 'RS':  # jump
                distance = z_transitions[i_zone + 1] - z_transitions[i_zone]
                if verbose:
                    print(("i={}, distance = {:2.2f} um".format(
                        i_zone, distance)))
                u_current.RS(z=distance,
                             n=refr_index_RS[i_zone],
                             kind='z',
                             fast=self.fast,
                             new_field=False,
                             verbose=False)
                # u_current.draw()
            else:  # genero experimento para BPM
                x0 = self.x
                z0 = np.linspace(z_transitions[i_zone],
                                 z_transitions[i_zone + 1], num_pixels_slice)
                u1 = mask_xz(x0, z0, u_current, self.wavelength,
                             self.n_background)
                # u1.draw_refraction_index(scale='equal')
                u1.BPM()
                u1.draw(draw_borders=False)
                fields_BPM.append(u1)
                u_current = u1.final_field()
        return u_current, fields_BPM, transitions

    def intensity(self):
        """Returns the intensity of the field

        Returns:
            (np.array): intensity of the field.
        """

        return np.abs(self.u)**2

    def check_intensity(self, draw=True, normalized=True):
        """
        Checks that intensity distribution is not lost by edges. It can be executed after a RS or BPM propagation.

        Parameters:
            draw (bool): Draws the intensity
            normalized (bool): Draws it normalized

        returns:
            (np.array): array with intensity I(z)
        """

        # intensity_prof = np.sum((np.abs(self.u)**2) * np.real(self.n), axis=0)
        intensity_prof = np.sum((np.abs(self.u)**2), axis=0)
        I_max = intensity_prof[0]
        if normalized is True:
            intensity_prof = intensity_prof / I_max
        if draw is True:
            plt.figure()
            plt.plot(self.z / mm, intensity_prof, 'k')
            plt.grid()
            plt.xlabel("$z\,(mm)$")
            plt.ylabel("$I(z)$")
            plt.ylim(bottom=0)

        return intensity_prof

    def detect_index_variations(self, n_edge, incr_n=0.1):
        """In a XZ masks, detects refraction index variations.

        Parameteres:
            n_edge (float):
            incr_n (float): refraction index variation to detect

        Returns:
            x_lens_l (np.array): x for left edge.
            h_lens_l (np.array): h for left edge.
            x_lens_r (np.array): x for right edge.
            h_lens_r (np.array): h for right edge.
        """
        z_new = self.z
        x_new = self.x

        iborders = np.real(self.n) > n_edge
        iborders = iborders + 0.0000001

        # surface detection
        diff1a = np.diff(iborders, axis=1)

        # cada uno de los lados
        ix_l, iz_l = (diff1a > incr_n).nonzero()
        ix_r, iz_r = (diff1a < -incr_n).nonzero()

        x_lens_l = x_new[ix_l]
        h_lens_l = z_new[iz_l]

        x_lens_r = x_new[ix_r]
        h_lens_r = z_new[iz_r]
        return x_lens_l, h_lens_l, x_lens_r, h_lens_r

    def _detect_transitions_(self, min_variation=1e-10):
        """Detects transitions areas and algorithms between RS and BPM.

        Parameters:
            min_variation (float): min index variation to detect

        Returns:
            (list floats) : z_transitions, positions z of transitions
            (list str) : algorithms, "RS" or "BPM"
            (list floats) : refr_index_RS, refraction indexes for RS
        """
        # para estar seguros que cogemos bien BPM y no empezamos tarde
        dz_bpm = 25 * um

        variation = np.std(np.abs(self.n), axis=0)

        z_transitions = [self.z[0]]
        num_transition = 0

        # comprobamos inicio si es RS o BPM
        if variation[0] < 1e-12:
            # print(("init {} - {}".format(variation[0], self.z[0])))
            algorithm = ['RS']
            refr_index_RS = [self.n[0, 0]]
        else:
            # print(("init {} - {}".format(variation[0], self.z[0])))
            algorithm = ['BPM']
            refr_index_RS = [-1]

        # print((self.z[1:]))
        for i, zi in enumerate(self.z[1:]):

            if i == len(self.z) - 2:  # for the last one if RS. store last one
                # print("a, the last one")
                # print((algorithm[-1]))
                # hay que calcular de todas formas, al menos una línea nueva
                num_transition = num_transition + 1
                z_transitions.append(zi)
                algorithm.append('RS')
                refr_index_RS.append(self.n[0, i])

            elif algorithm[
                    num_transition] == 'RS' and variation[i] > min_variation:
                # create new transition
                # print(("c {} - {} -> BPM".format(variation[i], self.z[i])))
                num_transition = num_transition + 1
                z_transitions.append(self.z[i] - dz_bpm)
                algorithm.append('BPM')
                refr_index_RS.append(-1)

            elif algorithm[
                    num_transition] == 'BPM' and variation[i] < min_variation:
                # create new transition
                # print(("d {} - {} -> RS".format(variation[i], self.z[i])))
                num_transition = num_transition + 1
                z_transitions.append(self.z[i] + dz_bpm)
                algorithm.append('RS')
                refr_index_RS.append(self.n[0, i])

        return z_transitions, algorithm, refr_index_RS

    def surface_detection(self,
                          mode=1,
                          min_incr=0.1,
                          reduce_matrix='standard',
                          has_draw=False):
        """detect edges of variation in refraction index.

        Parameters:
            mode (int): 1 or 2, algorithms for surface detection: 1-gradient, 2-diff
            min_incr (float): minimum incremental variation to detect
            reduce_matrix (int, int) or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
            has_draw (bool): If True draw.
        """

        if reduce_matrix is False:
            n_new = self.n
            z_new = self.z
            x_new = self.x

        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_z = len(self.z)
            reduction_x = int(num_x / 1000)
            reduction_z = int(num_z / 1000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_z == 0:
                reduction_z = 1

            n_new = self.n[::reduction_x, ::reduction_z]
            z_new = self.z[::reduction_z]
            x_new = self.x[::reduction_x]

        else:
            n_new = self.n[::reduce_matrix[0], ::reduce_matrix[1]]

            # cuidado, que puede ser al revés
            z_new = self.z[::reduce_matrix[1]]
            x_new = self.x[::reduce_matrix[0]]

        mode = 2
        if mode == 1:
            diff1 = gradient(np.abs(n_new), axis=0)
            diff2 = gradient(np.abs(n_new), axis=1)
        elif mode == 2:
            diff1 = diff(np.abs(n_new), axis=0)
            diff2 = diff(np.abs(n_new), axis=1)
            # print diff1.shape, diff2.shape, len(self.z), len(self.x)
            diff1 = np.append(diff1, np.zeros((1, len(z_new))), axis=0)
            diff2 = np.append(diff2, np.zeros((len(x_new), 1)), axis=1)
            # print diff1.shape, diff2.shape

        # if np.abs(diff1 > min_incr) or np.abs(diff2 > min_incr):
        t = np.abs(diff1) + np.abs(diff2)

        ix, iy = (t > min_incr).nonzero()
        self.borders = z_new[iy], x_new[ix]

        if has_draw:
            plt.figure()
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
            plt.imshow(t, extent=extension, alpha=0.5)

        return z_new[iy], x_new[ix]

    def draw(self,
             kind='intensity',
             logarithm=0,
             normalize='',
             draw_borders=False,
             filename='',
             scale='',
             min_incr=0.0005,
             reduce_matrix='standard',
             colorbar_kind=False,
             colormap_kind="",
             z_scale='um',
             edge_matrix=None,
             interpolation='spline36',
             **kwargs):
        """Draws  XZ field.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'real'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: incrimum increment in refraction index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
            z_scale (str): 'mm', 'um'
            edge_matrix (numpy.array): positions of borders
            interpolation(str): methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        """

        if reduce_matrix is False:
            amplitude, intensity, phase = field_parameters(self.u, True)

        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_z = len(self.z)
            reduction_x = int(num_x / 2000)
            reduction_z = int(num_z / 2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_z == 0:
                reduction_z = 1
            u_new = self.u[::reduction_x, ::reduction_z]
            amplitude, intensity, phase = field_parameters(u_new, True)
        else:
            u_new = self.u[::reduce_matrix[0], ::reduce_matrix[1]]
            amplitude, intensity, phase = field_parameters(u_new, True)

        # phase[intensity < 0.005 * (intensity.max())] = 0

        if z_scale == 'um':
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
        elif z_scale == 'mm':
            extension = [
                self.z[0] / mm, self.z[-1] / mm, self.x[0], self.x[-1]
            ]

        percentaje_intensity = params_drawing['percentaje_intensity']

        plt.figure()

        if kind == 'intensity':
            I_drawing = intensity
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'amplitude':
            I_drawing = amplitude
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'phase':
            phase = phase / degrees
            phase[intensity < percentaje_intensity * (intensity.max())] = 0

            I_drawing = phase
        elif kind == 'real':
            I_drawing = np.real(self.u)
        else:
            print("bad kind parameter")
            return

        h1 = plt.imshow(I_drawing,
                        interpolation=interpolation,
                        aspect='auto',
                        origin='lower',
                        extent=extension, **kwargs)

        if z_scale == 'um':
            plt.xlabel('z ($\mu m$)')
        elif z_scale == 'mm':
            plt.xlabel('z (mm)')

        plt.ylabel('x ($\mu m$)')

        if colormap_kind in ('', [], None, True):
            if kind == 'intensity':
                colormap_kind = self.params_drawing["color_intensity"]
            if kind == 'amplitude':
                colormap_kind = self.params_drawing["color_amplitude"]
            if kind == 'phase':
                colormap_kind = self.params_drawing["color_phase"]
            if kind == 'real':
                colormap_kind = self.params_drawing["color_real"]

        if kind == 'intensity':
            climits = I_drawing.min(), I_drawing.max()
        if kind == 'amplitude':
            climits = -I_drawing.max(), I_drawing.max()
        if kind == 'phase':
            climits = -180, 180
        if kind == 'real':
            climits = -I_drawing.max(), I_drawing.max()

        plt.axis(extension)

        if colorbar_kind not in (False, '', [], None):
            plt.colorbar(orientation=colorbar_kind)

        h1.set_cmap(colormap_kind)  # OrRd # Reds_r gist_heat
        plt.clim(climits)

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if edge_matrix is not None:
                border0, border1 = edge_matrix
            else:
                if self.borders is None:
                    self.surface_detection(1, min_incr, reduce_matrix)
                border0 = self.borders[0]
                border1 = self.borders[1]

            plt.plot(border0, border1, 'w.', ms=0.5)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_refraction_index(self,
                              kind='all',
                              draw_borders=True,
                              title='',
                              filename='',
                              scale='',
                              min_incr=0.01,
                              reduce_matrix='standard',
                              colorbar_kind=None,
                              colormap_kind=cm.Blues,
                              edge_matrix=None):
        """Draws refraction index.

        Parameters:
            kind (str): 'all', 'real', 'imag'
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            title (str): title of drawing
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: minimum increment in refraction index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
            edge_matrix (numpy.array): positions of borders
        """

        plt.figure()
        extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

        if kind == 'all':
            n_draw = np.abs(self.n)
        elif kind == 'real':
            n_draw = np.real(self.n)
            n_draw[np.abs(np.imag(self.n)) > 0] = self.n_background
        elif kind == 'imag':
            n_draw = np.imag(self.n)

        if reduce_matrix is False:
            h1 = plt.imshow(n_draw,
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_z = len(self.z)
            reduction_x = int(num_x / 2000)
            reduction_z = int(num_z / 2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_z == 0:
                reduction_z = 1
            n_new = n_draw[::reduction_z, ::reduction_x]
            h1 = plt.imshow(n_new,
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)
        else:
            n_new = n_draw[::reduce_matrix[0], ::reduce_matrix[1]]
            h1 = plt.imshow(n_new,
                            interpolation='bilinear',
                            aspect='auto',
                            origin='lower',
                            extent=extension)

        plt.xlabel('z ($\mu m$)')
        plt.ylabel('x ($\mu m$)')
        plt.title(title)

        plt.axis(extension)
        h1.set_cmap(colormap_kind)  # flag OrRd # Reds_r gist_heat # gist_heat

        if colorbar_kind not in (False, '', None):
            plt.colorbar(orientation=colorbar_kind)

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if self.borders is None or edge_matrix is None:
                self.surface_detection(1, min_incr, reduce_matrix)
                border0 = self.borders[0]
                border1 = self.borders[1]
            if edge_matrix is not None:
                border0, border1 = edge_matrix
            plt.plot(border0, border1, 'w.', ms=1)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_incident_field(self,
                            kind='intensity',
                            logarithm=False,
                            normalize=False,
                            filename=''):
        """Draws incident field self.u0

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase', 'fill', 'fft'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            filename (str): if not '' stores drawing in file,
        """

        u_inc = Scalar_field_X(x=self.x,
                               wavelength=self.wavelength,
                               n_background=1,
                               info="incident_field")
        u_inc.u = self.u0.u
        u_inc.draw(kind, logarithm, normalize, None, filename)

    def profile_longitudinal(self,
                             kind='intensity',
                             x0=0 * um,
                             logarithm=False,
                             normalize=False,
                             draw=True,
                             filename=''):
        """Determine and draws longitudinal profile

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'refraction_index'
            x0 (float): profile that passes through x=x0
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file

        Returns:
            numpy.array: profile
        """

        imenor, value, distance = nearest(vector=self.x, number=x0)

        if kind == 'refraction_index':
            n_profile = np.abs(self.n[imenor, :])
            I_drawing = n_profile
        else:
            u = np.squeeze(self.u[imenor, :])
            I_drawing = prepare_drawing(u, kind, logarithm, normalize)
            amplitude, intensity, phase = field_parameters(u)

        if draw is True:
            plt.figure(facecolor='w', edgecolor='k')
            plt.plot(self.z, I_drawing, 'k', linewidth=2)  # 'k-o'
            plt.axis([self.z[0], self.z[-1], I_drawing.min(), I_drawing.max()])

            texto = 'I(z, x=%d um)' % (x0)
            plt.xlabel('z (um)')
            plt.ylabel(texto)

            if not filename == '':
                plt.savefig(filename,
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1)

        if kind == 'intensity':
            output = intensity
        elif kind == 'amplitude':
            output = amplitude
        elif kind == 'phase':
            output = phase
        elif kind == 'refraction_index':
            output = n_profile
        else:
            output = None
        return output

    def profile_transversal(self,
                            kind='intensity',
                            z0=0 * um,
                            logarithm=False,
                            normalize=False,
                            draw=True,
                            filename=''):
        """Determine and draws transversal profile.

        Parameters:
            kind (str): type of drawing:  'amplitude', 'intensity', 'phase', 'refraction_index'
            z0 (float): profile that passes through z=z0
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file,

        Returns:
            numpy.array: profile

        TODO:
            Include interpolation
        """

        imenor, value, distance = nearest(vector=self.z, number=z0)

        if kind == 'refraction_index':
            n_profile = np.abs(self.n[:, imenor])
            I_drawing = n_profile
        else:
            u = np.squeeze(self.u[:, imenor])
            I_drawing = prepare_drawing(u, kind, logarithm, normalize)
            amplitude, intensity, phase = field_parameters(u)

        if draw is True:
            plt.figure(facecolor='w', edgecolor='k')
            plt.plot(self.x, I_drawing, 'k', linewidth=2)  # 'k-o'
            plt.axis([self.x[0], self.x[-1], I_drawing.min(), I_drawing.max()])
            texto = 'I(z=%d um, x)' % (z0)
            plt.xlabel('x (um)')
            plt.ylabel(texto)

            if not filename == '':
                plt.savefig(filename,
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1)
                # ,dpi=600,bbox_inches='tight',pad_inches=0

        if kind == 'intensity':
            output = intensity
        elif kind == 'amplitude':
            output = amplitude
        elif kind == 'phase':
            output = phase
        elif kind == 'refraction_index':
            output = n_profile
        else:
            output = None
        return output

    def search_focus(self, verbose=True):
        """Search for location of maximum.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'refraction_index'
            x0 (float): profile that passes through x=x0
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw (bool): If True, draws, False only returns profile
            filename (str): if not '' stores drawing in file,

        Returns:
            (x,z): positions of focus
        """
        intensity = np.abs(self.u)**2
        # busca el máximo de una matrix bidimensional
        ix, iz = np.unravel_index(intensity.argmax(), intensity.shape)
        if verbose is True:
            print(("x = {:2.3f} um, z = {:2.3f} um".format(
                self.x[ix], self.z[iz])))
        return self.x[ix], self.z[iz]

    def beam_widths(self, kind='FWHM1D',
                    has_draw=[True, False],
                    percentaje=0.5,
                    remove_background=None,
                    verbose=False):
        """Computes the beam width for all the distances z.
        Parameters:
            kind (str): kind of algorithm: 'sigma4', 'FWHM1D'
            has_draw (bool, bool): First for complete analysis, second for all FWHM2D computations
            verbose (bool): prints info

        Returns:
            (numpy.array) widths:  for each distance z
            (numpy.array) positions_center: positions of centers for each z
        """

        widths = np.zeros_like(self.z)
        positions_center = np.zeros_like(self.z)

        for i, zi in enumerate(self.z):
            field = np.abs(self.u[:, i])
            if kind == 'sigma4':
                widths[i], positions_center[i] = beam_width_1D(field, self.x)
            elif kind == 'FWHM1D':
                intensity = np.abs(field)**2
                widths[i] = FWHM1D(self.x,
                                   intensity,
                                   percentaje=percentaje,
                                   remove_background=remove_background,
                                   has_draw=has_draw[1])

        if has_draw[0] is True:
            plt.figure()
            plt.plot(self.z, widths, 'r', label='axis')
            plt.xlabel("z ($\mu$m)")
            plt.ylabel("widths ($\mu$m)")
            plt.legend()

        return widths, positions_center

    def video_profiles(self,
                       kind='intensity',
                       kind_profile='transversal',
                       step=1,
                       wait=0.001,
                       logarithm=False,
                       normalize=False,
                       filename='',
                       verbose=False):
        """Draws profiles in a video fashion

        Parameters:
            kind (str): 'intensity', 'amplitude', 'phase'
            kind_profile (str): 'transversal', 'longitudinal'
            step (list): number of frames shown (if 1 shows all, if 2 1/2, ..) for accelerating pruposes in video.
            wait (float) : (in seconds) time for slow down the video
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            filename: (str))  filename of video
            verbose (bool): If True shows info
        """

        fig = plt.figure()
        if kind_profile == 'transversal':
            h1, = plt.plot(self.x, np.zeros_like(self.x), 'k', lw=2)
            plt.xlim(self.x[0], self.x[-1])
            plt.xlabel(r'$x (\mu m)$')
        elif kind_profile == 'longitudinal':
            h1, = plt.plot(self.z, np.zeros_like(self.z), 'k', lw=2)
            plt.xlim(self.z[0], self.z[-1])
            plt.xlabel(r'$z (\mu m)$')

        I_drawing = prepare_drawing(self.u, kind, logarithm, normalize)

        plt.ylim(I_drawing.min(), I_drawing.max())

        writer = prepare_video(fps=15, title='', artist='', comment='')

        with writer.saving(fig, filename, 300):
            if kind_profile == 'transversal':
                for i in range(0, len(self.z), step):
                    h1.set_ydata(I_drawing[:, i])
                    plt.title("z={:6.2f}, i={}".format(round(self.z[i], 2), i))
                    plt.draw()
                    if filename == '':
                        plt.pause(wait)
                    else:
                        if verbose:
                            print(("{}/{}".format(i, len(self.z))))
                        writer.grab_frame()
            elif kind_profile == 'longitudinal':
                for i in range(0, len(self.x), step):
                    h1.set_ydata(I_drawing[i, :])
                    plt.title("x={:6.2f}, i={}".format(round(self.x[i], 2), i))
                    plt.draw()
                    if filename == '':
                        plt.pause(wait)
                    else:
                        if verbose:
                            print(("{}/{}".format(i, len(self.z))))
                        writer.grab_frame()
        plt.close('')

    def video(self,
              kind='intensity',
              z_min=None,
              z_max=None,
              logarithm=False,
              normalize=False,
              time_video=10 * seconds,
              frames_reduction=5,
              filename='video.avi',
              dpi=300):
        """Generates a video in the z dimension.

        Parameters:
            kind (str):
            z_min (float):
            z_max (float):
            logarithm (bool):
            normalize (bool):
            time_video (float):
            frames_reduction (int):
            filename (str):
            dpi (int):
        """

        I_drawing = prepare_drawing(self.u, kind, logarithm, normalize)
        if z_min is None:
            z_min = self.z[0]
        if z_max is None:
            z_max = self.z[-1]

        imin, _, _ = nearest(self.z, z_min)
        imax, _, _ = nearest(self.z, z_max)

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.grid()
        hdl_line, = ax.plot([], [], 'k', lw=2)
        ax.set_title('', transform=ax.transAxes)
        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(I_drawing.min(), I_drawing.max())

        def init():
            hdl_line.set_data([], [])
            ax.set_title('')
            return hdl_line

        def animate(i):

            hdl_line.set_data(self.x, I_drawing[:, i])
            ax.set_title("$z = {:2.0f} \mu m$".format(self.z[i]))
            return i

        ani = animation.FuncAnimation(fig,
                                      animate,
                                      list(range(imin, imax,
                                                 frames_reduction)),
                                      interval=25,
                                      blit=False,
                                      init_func=init)

        fps = int(len(self.z) / (time_video * frames_reduction))

        ani.save(filename, fps=fps, dpi=dpi)
        plt.close()

    def draw_profiles_interactive(self,
                                  kind='intensity',
                                  logarithm=False,
                                  normalize=False):
        """Draws profiles interactivey. Only transversal

        Parameters:
            kind (str): 'intensity', 'amplitude', 'phase'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
        """

        global l2a, zZ, I_drawing, z, h1, x, log1, norm1
        plt.figure()
        h1, = plt.plot([self.z[0], self.z[0]], [self.x[0], self.x[-1]],
                       lw=2,
                       color='w')

        I_drawing = prepare_drawing(self.u, kind, logarithm, normalize)

        log1 = logarithm
        norm1 = normalize

        z_actual = self.z[0]
        x = self.x
        ix, iz = np.unravel_index(I_drawing.argmax(), I_drawing.shape)
        # I_drawing_maxima = I_drawing[ix, iz]
        extension = [self.x[0], self.x[-1], I_drawing.min(), I_drawing.max()]

        imenor, value, distance = nearest(vector=self.z, number=z_actual)
        I_drawing_actual = np.squeeze(I_drawing[:, imenor])

        z = self.z

        plt.subplots_adjust(left=0.15, bottom=0.25)
        plt.axis(extension)
        l2a, = plt.plot(self.x, I_drawing_actual, lw=2, color='k')
        axcolor = 'lightgoldenrodyellow'
        axS = plt.axes([0.15, 0.15, 0.75, 0.03], facecolor=axcolor)
        zZ = plt.Slider(axS,
                        'z (um)',
                        self.z[0],
                        self.z[-1],
                        valinit=self.z.min())
        zZ.on_changed(__update__)


def __update__(val):
    """for making videos.
    """
    zz = zZ.val
    imenor, value, distance = nearest(vector=z, number=zz)
    I_drawing_profile = np.squeeze(I_drawing[:, imenor])
    # lo siguiente normaliza linea a línea:
    I_drawing_profile = normalize_draw(I_drawing_profile, log1, norm1)
    l2a.set_ydata(I_drawing_profile)
    plt.draw()

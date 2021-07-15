# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_field_XYZ class and several functions for multiprocessing.
It is required also for generating masks and fields.
The main atributes are:
    * self.u - field XYZ
    * self.x - x positions of the field
    * self.y - y positions of the fieldalgorithm
    * self.z - z positions of the field
    * self.wavelength - wavelength of the incident field. The field is monochromatic
    * self.u0 - initial field at z=z0
    * self.n_background - background refraction index
    * self.n - refraction index


The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar fields*

*Definition of a scalar field*
    * load and save data
    * to_Scalar_field_XY
    * xy_2_xyz
    * cut_function
    * __rotate__
    * __rotate_axis__

*Propagation*
    * RS - Rayleigh Sommerfeld
    * RS_amplification
    * BPM - Beam Propagation method

*Drawing functions*
    * draw_XYZ
    * draw_XY
    * draw_XZ
    * draw_volume
    * draw_refraction_index
    * video

"""
import copyreg
import os
import sys
import time
import types
from multiprocessing import Pool

import matplotlib.animation as anim
from numpy import cos, diff, gradient, sin
from scipy.fftpack import fft2, ifft2
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.signal import correlate2d

from . import degrees, eps, mm, np, num_max_processors, params_drawing, plt
from .scalar_fields_XY import PWD_kernel, Scalar_field_XY, WPM_schmidt_kernel
from .scalar_fields_XZ import Scalar_field_XZ
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw, prepare_drawing
from .utils_math import get_k, ndgrid, nearest
from .utils_multiprocessing import _pickle_method, _unpickle_method
from .utils_optics import FWHM2D, beam_width_2D, field_parameters

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class Scalar_field_XYZ(object):
    """Class for 3D scalar fields.

    Parameters:
        u0 (Scalar_field_XY): Initial scalar field. wavelength, and x, y arrays are obtained from this field.
        z (numpy.array): linear array with equidistant positions.
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.u (numpy.array): equal size than X. complex field
        self.wavelength (float): wavelength of the incident field.
        self.u0 (Scalar_field_XY): Initial XY field
        self.n_background (float): background refraction index.
        self.n (numpy.array): refraction index. Same dimensions than self.u.
    """
    def __init__(self,
                 x=None,
                 y=None,
                 z=None,
                 wavelength=None,
                 n_background=1.,
                 info=''):

        self.x = x
        self.y = y
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        self.fast = True
        self.quality = 0
        self.borders = None
        self.params_drawing = params_drawing

        if x is not None and z is not None:
            self.X, self.Y, self.Z = ndgrid(x, y, z)
            self.u0 = Scalar_field_XY(x, y, wavelength)
            self.u = np.zeros_like(self.X, dtype=complex)
            self.n = n_background * np.ones_like(self.X, dtype=complex)
        else:
            self.X = None
            self.Y = None
            self.Z = None
            self.u0 = None
            self.u = None
            self.n = None
        self.info = info
        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Scalar_field_XYZ'
        self.date = get_date()

    def xy_2_xyz(self, u0_XY, z):
        """Similar to Init. send a Scalarfield_XY and passes to XYZ.

        Parameters:
            u0_XY (Scalar_field_XY): init field
            z (numpy.array): array with z positions
        """
        u0 = u0_XY[0]
        self.x = u0.x
        self.y = u0.y
        self.z = z
        # la longitud de onda tiene que ir en cada Scalar_source_XY,
        # pues podriamos tener la suma de dos fuentes con diferente lambda
        self.wavelength = u0.wavelength
        self.u0 = u0
        self.amplification = 1
        self.X, self.Y, self.Z = ndgrid(self.x, self.y, self.z)

        self.u = np.zeros_like(self.X, dtype=complex)
        self.n = np.ones(np.shape(self.X),
                         dtype=float)  # el índice de refracción

        for i in range(len(self.z)):
            # print self.u[:,:,i].shape
            # print Scalar_field_XY[i].u.shape
            self.u[:, :, i] = u0_XY[i].u

    def __str__(self):
        """Represents main data."""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - x:  {},   y:  {},  z:  {},   u:  {}".format(
            self.type, self.x.shape, self.y.shape, self.z.shape, self.u.shape))
        print(" - xmin:       {:2.2f} um,  xmax:      {:2.2f} um".format(
            self.x[0], self.x[-1]))
        print(" - ymin:       {:2.2f} um,  ymax:      {:2.2f} um".format(
            self.y[0], self.y[-1]))
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
        """Adds two Scalar_field_XYZ. For example two light sources or two masks.

        Parameters:
            other (Scalar_field_XYZ): 2nd field to add
            kind (str): instruction how to add the fields: - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1. - 'standard': add fields u3=u1+u2 and does nothing.

        Returns:
            Scalar_field_XYZ: `u3 = u1 + u2`
        """

        u3 = Scalar_field_XYZ(self.x, self.y, self.z, self.wavelength,
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
        """Substract two Scalar_field_XYZ For example two light sources or two masks.

        Parameters:
            other (Scalar_field_XYZ): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        # TODO:
            It can be improved for maks (not having less than 1)
        """

        u3 = Scalar_field_XYZ(self.x, self.y, self.z, self.wavelength,
                              self.n_background)
        u3.n = self.n
        u3.u = self.u - other.u
        return u3

    def __rotate__(self, psi, phi, sigma):
        """Function to rotate around any of the 3 axis of rigid solid.

        Parameters:
            psi (float): Euler angle in radians
            phi (float): Euler angle in radians
            sigma (float): Euler angle in radians

        Returns:
            numpy.array: Yrot: (xyz matrix rotation of solid angle)
            numpy.array: Xrot: (xyz matrix rotation of solid angle)
            numpy.array: Zrot: (xyz matrix rotation of solid angle)

        References:
            http://estudiarfisica.wordpress.com/2011/03/17/ampliacion-del-solido-rigido-matrices-de-rotation-angles-y-transformaciones-de-euler-velocidad-angular-momento-angular-tensor-de-inercia-teorema-de-steiner-utilsizado/
        """
        cp = cos(psi)
        sp = sin(psi)
        cf = cos(phi)
        sf = cos(phi)
        cs = cos(sigma)
        ss = sin(sigma)

        Xrot = self.X * (cp * cf - sp * cs * sf) + self.Y * (
            cp * sf + sp * cs * cf) + self.Z * (sp * ss)
        Yrot = self.X * (-sp * cf - cp * cs * sf) + self.Y * (
            -sp * sf + cp * cs * cf) + self.Z * (cp * ss)
        Zrot = self.X * (ss * sf) + self.Y * (-ss * cf) + self.Z * (cs)
        return Xrot, Yrot, Zrot

    def __rotate_axis__(self, axis, angle):
        """rotate around an axis.

        Parameters:
            axis (float, float, float): direction of the axis
            angle (float): angle of rotation in radians
            sigma (float): Euler angle in radians

        Returns:
            numpy.array: Yrot: direction of the axis
            numpy.array: Xrot: (xyz matrix rotation of solid angle)
            numpy.array: Zrot: (xyz matrix rotation of solid angle)
        """
        # normalized axis
        u, v, w = axis / np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)

        ct = cos(angle)
        st = sin(angle)

        Xrot = self.X * (u**2 + (v**2 + w**2) * ct) + self.Y * (
            u * v * (1 - ct) - w * st) + self.Z * (u * w * (1 - ct) + v * st)
        Yrot = self.X * (u * v * (1 - ct) + w * st) + self.Y * (
            v**2 + (u**2 + w**2) * ct) + self.Z * (v * w * (1 - ct) - u * st)
        Zrot = self.X * (u * w * (1 - ct) - v * st) + self.Y * (
            v * w * (1 - ct) + u * st) + self.Z * (w**2 + (u**2 + v**2) * ct)

        return Xrot, Yrot, Zrot

    def clear_field(self):
        """clear field."""

        self.u = np.zeros(np.shape(self.u), dtype=complex)

    def clear_refraction_index(self):
        """clear refraction index n(x,z)=n_background."""

        self.n = self.n_background * np.ones_like(self.X, dtype=complex)

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

        if verbose:
            print(dict0.keys())

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

    def cut_resample(self,
                     x_limits='',
                     y_limits='',
                     z_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind=(3, 1)):
        """it cut the field to the range (x0,x1). If one of this x0,x1 positions is out of the self.x range it do nothing. It is also valid for resampling the field, just write x0,x1 as the limits of self.x

        Parameters:
            x_limits (float,float): (x0,x1) starting and final points to cut, if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut, if '' - takes the current limit y[0] and z[-1]
            num_points (int): it resamples x, y and u, where [],'',0,None -> it leave the points as it is new_field (bool): it returns a new Scalar_field_XY
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if z_limits == '':
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)

        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y, num_points_z = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            z_new = np.linspace(z0, z1, num_points_z)
            field_n = Scalar_field_XYZ(x=x_new,
                                       y=y_new,
                                       z=z_new,
                                       wavelength=self.wavelength,
                                       n_background=self.n_background)

            X_new = field_n.X
            Y_new = field_n.Y
            Z_new = field_n.Z

            f_interp_amplitude = RegularGridInterpolator(
                (self.x, self.y, self.z), np.abs(self.u))
            f_interp_phase = RegularGridInterpolator((self.x, self.y, self.z),
                                                     np.angle(self.u))
            u_new_abs = f_interp_amplitude((X_new, Y_new, Z_new))
            u_new_phase = f_interp_phase((X_new, Y_new, Z_new))
            u_new = u_new_abs * np.exp(1j * u_new_phase)

            n_interp_real = RegularGridInterpolator((self.x, self.y, self.z),
                                                    np.real(self.n))
            n_interp_imag = RegularGridInterpolator((self.x, self.y, self.z),
                                                    np.imag(self.n))
            n_new_real = n_interp_real((X_new, Y_new, Z_new))
            n_new_imag = n_interp_imag((X_new, Y_new, Z_new))
            n_new = n_new_real + 1j * n_new_imag

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            k_s = slice(i_z0, i_z1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            z_new = self.z[k_s]
            X_new, Y_new, Z_new = ndgrid(x_new, y_new, z_new)
            u_new = self.u[i_s, j_s, k_s]
            n_new = self.n[i_s, j_s, k_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.z = z_new

            self.X = X_new
            self.Y = Y_new
            self.Z = Z_new

            self.u = u_new
            self.n = n_new

        elif new_field is True:
            field_n = Scalar_field_XYZ(x=x_new,
                                       y=y_new,
                                       z=z_new,
                                       wavelength=self.wavelength,
                                       n_background=self.n_background)
            field_n.u = u_new
            field_n.n = n_new
            return field_n

    def incident_field(self, u0, z0=None):
        """Incident field for the experiment. It takes a Scalar_source_XYZ field.

        Parameters:
            u0 (Scalar_source_X): field produced by Scalar_source_XYZ (or a XYZ field)
            z0 (float): position of the incident field. if None, '', [], is at the beginning
        """

        self.u0 = u0

        if z0 in (None, '', []):
            self.u[:, :, 0] = self.u[:, :, 0] + u0.u.transpose()

        else:
            iz, _, _ = nearest(self.z, z0)
            self.u[:, :, iz] = self.u[:, :, iz] + u0.u.tranpose()

    def final_field(self):
        """Returns the final field as a Scalar_field_XYZ."""

        u_final = Scalar_field_XY(x=self.x,
                                  y=self.y,
                                  wavelength=self.wavelength,
                                  n_background=1,
                                  info="from final_field at z0= {} um".format(
                                      self.z[-1]))
        u_final.u = self.u[:, :, -1]
        return u_final

    def __RS_multiprocessing__(self, i):
        """Internal for multiprocessing.

        Parameters:
            i (int): Number for for loop.
        """
        self.u[:, :, i] = self.u0.RS(amplification=(1, 1),
                                     z=self.z[i],
                                     n=self.n_background,
                                     new_field=False,
                                     matrix=True,
                                     kind='z',
                                     verbose=False)
        return self.u[:, :, i]

    def RS(self, verbose=False, num_processors=num_max_processors - 2):
        """Rayleigh Sommerfeld propagation algorithm

        Parameters:
            verbose (bool): shows the quality of algorithm (>1 good)
            num_processors (int): number of processors for multiprocessing

        Returns:
           time in the processing
        """

        time1 = time.time()
        if num_processors == 1:
            for iz in np.array(range(0, len(self.z))):
                self.u[:, :, iz] = self.u0.RS(amplification=(1, 1),
                                              z=self.z[iz],
                                              n=self.n_background,
                                              new_field=False,
                                              matrix=True,
                                              kind='z',
                                              verbose=verbose)

        else:
            pool = Pool(num_processors)
            t = pool.map(self.__RS_multiprocessing__, list(range(len(self.z))))
            pool.close()
            pool.join()
            for i in range(len(self.z)):
                self.u[:, :, i] = t[i]

        time2 = time.time()
        print(("time in RS= {}. num proc= {}".format(time2 - time1,
                                                     num_processors)))
        return time2 - time1

    def RS_amplification(self, amplification=(1, 1)):
        """Rayleigh Sommerfeld propagation algorithm. it performs amplification

        Parameters:
            amplification (int,int): number of fields

        Returns:
           Scalar_field_XY:
        """

        x0 = self.x[0]
        y0 = self.y[0]
        x1 = self.x[-1]
        y1 = self.y[-1]
        nx = len(self.x)
        ny = len(self.y)

        Gx = np.linspace(x0 * amplification, x1 * amplification,
                         nx * amplification)
        Gy = np.linspace(y0 * amplification, y1 * amplification,
                         ny * amplification)

        field_output = Scalar_field_XY(Gx, Gy, self.wavelength)

        fieldij = Scalar_field_XYZ(self.x, self.y, self.wavelength)
        for ix in range(0, amplification):
            ixini = ix * nx
            ixfin = (ix + 1) * nx - 1
            xini = Gx[ixini]
            xfin = Gx[ixfin]
            for iy in range(0, amplification):
                iyini = iy * ny
                iyfin = (iy + 1) * ny - 1
                yini = Gy[iyini]
                yfin = Gy[iyfin]

                xij = np.linspace(xini, xfin, nx)
                yij = np.linspace(yini, yfin, ny)

                fieldij.propagacionRS(xij, yij)
                # quality=min(quality,parametros.quality);
                field_output[ixini:ixfin, iyini:iyfin] = fieldij.u

            # parametros.tiempo=tiempoTotal
            # parametros.quality=quality
        return field_output

    def BPM(self, has_edges=True, pow_edge=80, verbose=False):
        """3D Beam propagation method (BPM).

        Parameters:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            verbose (bool): shows data process by screen


        References:
            Algorithm in "Engineering optics with matlab" pag 119
        """

        k0 = 2 * np.pi / self.wavelength
        numz = len(self.z)
        numx = len(self.x)
        numy = len(self.y)

        deltaz = self.z[1] - self.z[0]
        rangox = self.x[-1] - self.x[0]
        rangoy = self.y[-1] - self.y[0]
        # pixelx = np.linspace(-int(numx / 2), int(numx / 2), numx)
        # pixely = np.linspace(-numy / 2, numy / 2, numy)

        modo = self.u0.u

        kx1 = np.linspace(0, int(numx / 2) + 1, int(numx / 2))
        kx2 = np.linspace(-int(numx / 2), -1, int(numx / 2))
        kx = (2 * np.pi / rangox) * np.concatenate((kx1, kx2))

        ky1 = np.linspace(0, numy / 2 + 1, int(numy / 2))
        ky2 = np.linspace(-numy / 2, -1, int(numy / 2))
        ky = (2 * np.pi / rangoy) * np.concatenate((ky1, ky2))

        KX, KY = ndgrid(kx, ky)

        phase1 = np.exp((1j * deltaz * (KX**2 + KY**2)) / (2 * k0))
        field = np.zeros(np.shape(self.n), dtype=complex)

        if has_edges:
            gaussX = np.exp(-(self.X[:, :, 0] / (self.x[0]))**pow_edge)
            gaussY = np.exp(-(self.Y[:, :, 0] / (self.y[0]))**pow_edge)
            filter_edge = (gaussX * gaussY)

        field[:, :, 0] = modo
        for k in range(0, numz):
            if verbose is True:
                print("BPM 3D: {}/{}".format(k, numz), end="\r")
            phase2 = np.exp(-1j * self.n[:, :, k] * k0 * deltaz)
            # Aplicamos la Transformada Inversa
            modo = ifft2((fft2(modo) * phase1)) * phase2
            if has_edges:
                modo = modo * filter_edge
            field[:, :, k] = modo
            self.u = field

    def PWD(self, n=None, matrix=False, verbose=False):
        """
        Plane wave decompostion algorithm.

        Parameters:
            n (np. array): refraction index, If None, it is n_background
            verbose (bool): If True prints state of algorithm

        Returns:
            numpy.array(): Field at at distance dz from the incident field

        References:
            1. Schmidt, S. et al. Wave-optical modeling beyond the thin-element-approximation. Opt. Express 24, 30188 (2016).


        Todo:
            include filter for edges
        """

        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        dz = self.z[1] - self.z[0]
        k0 = 2 * np.pi / self.wavelength

        if n is None:
            n = self.n_background

        kx = get_k(self.x, '+')
        ky = get_k(self.y, '+')
        Kx, Ky = np.meshgrid(kx, ky)
        K_perp2 = Kx**2 + Ky**2

        self.clear_field()
        num_steps = len(self.z)

        self.u[:, :, 0] = self.u0.u
        for i, zi in enumerate(self.z[0:-1]):
            result = self.u[:, :, i]
            result_next = PWD_kernel(result, n, k0, K_perp2, dz)
            self.u[:, :, i + 1] = result_next
            if verbose is True:
                print("{}/{}".format(i, num_steps), end='\r')

        if matrix is True:
            return self.u

    def WPM(self, has_edges=True, pow_edge=80, verbose=False):
        """
        WPM Methods.
        'schmidt method is very fast, only needs discrete number of refraction indexes'


        Parameters:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength
        x = self.x
        y = self.y
        z = self.z
        # dx = x[1] - x[0]
        # dy = y[1] - y[0]
        dz = z[1] - z[0]

        self.u[:, :, 0] = self.u0.u
        kx = get_k(x, flavour='+')
        ky = get_k(y, flavour='+')

        KX, KY = np.meshgrid(kx, ky)

        k_perp2 = KX**2 + KY**2
        # k_perp = np.sqrt(k_perp2)

        if has_edges is False:
            filter_edge = 1
        else:
            gaussX = np.exp(-(self.X[:, :, 0] / (self.x[0]))**pow_edge)
            gaussY = np.exp(-(self.Y[:, :, 0] / (self.y[0]))**pow_edge)
            filter_edge = (gaussX * gaussY)

        t1 = time.time()

        num_steps = len(self.z)
        for j in range(1, num_steps):

            self.u[:, :, j] = self.u[:, :, j] + WPM_schmidt_kernel(
                self.u[:, :, j - 1], self.n[:, :, j - 1], k0, k_perp2,
                dz) * filter_edge

            if verbose is True:
                if sys.version_info.major == 3:
                    print("{}/{}".format(j, num_steps), sep='\r', end='\r')
                else:
                    print("{}/{}".format(j, num_steps))

        t2 = time.time()

        if verbose is True:
            print("Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                t2 - t1, (t2 - t1) / len(self.z) * 1000))

    # def M_xyz(self, j, kx, ky):
    #     """
    #     TODO: sin terminar
    #     Refraction matrix given in eq. 18 from  M. W. Fertig and K.-H. Brenner,
    #     “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.
    #     """
    #     # simple parameters
    #
    #     k0 = 2 * np.pi / self.wavelength
    #
    #     z = self.z
    #     x = self.x
    #     y = self.y
    #
    #     num_x = self.x.size
    #     num_y = self.y.size
    #     num_z = self.z.size
    #
    #     dz = z[1] - z[0]
    #     dx = x[1] - x[0]
    #     dx = y[1] - y[0]
    #
    #     nj = self.n[:, j]
    #     nj_1 = self.n[:, j - 1]
    #
    #     n_med = nj.mean()
    #     n_1_med = nj_1.mean()
    #
    #     # parameters
    #     NJ, KX = np.meshgrid(nj, kx)
    #     NJ_1, KX = np.meshgrid(nj_1, kx)
    #
    #     k_perp = KX + eps
    #
    #     kz_j = np.sqrt((n_med * k0)**2 - k_perp**2)
    #     kz_j_1 = np.sqrt((n_1_med * k0)**2 - k_perp**2)
    #
    #     tg_TM = 2 * NJ_1**2 * kz_j / (NJ**2 * kz_j_1 + NJ_1**2 * kz_j)
    #     t_TE = 2 * kz_j_1 / (kz_j_1 + kz_j)
    #
    #     kj_1 = NJ_1 * k0
    #     fj_1 = k_perp**2 / (NJ_1**2 * kj_1**2)
    #     # cuidado no es la misma definición, me parece que está repetido
    #     e_xj_1 = np.gradient(NJ_1**2, dx, axis=0)
    #     eg_xj_1 = fj_1 * e_xj_1
    #     # cuidado no es la misma definición, me parece que está mal por no ser simétrica
    #
    #     p001 = 0
    #     p002 = tg_TM * (1 - 1j * eg_xj_1)
    #     p111 = t_TE
    #     p112 = 0
    #
    #     # M00
    #     M00 = (p001 + p002)
    #
    #     # M01
    #     M01 = 0
    #
    #     # M10
    #     M10 = 0
    #
    #     # M11
    #     M11 = (p111 + p112)
    #
    #     return M00, M01, M10, M11

    def to_Scalar_field_XY(self,
                           iz0=None,
                           z0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Scalar_field_XY. Only one of the first two variables (iz0,z0) should be used

        Parameters:
            iz0 (int): position i of z data in array
            z0 (float): position z to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        TODO:
            Simplify and change variable name clase
        """
        if is_class is True:
            field_output = Scalar_field_XY(x=self.x,
                                           y=self.y,
                                           wavelength=self.wavelength)
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
            field_output.u = np.squeeze(self.u[:, :, iz])
            return field_output

        if matrix is True:
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
            return np.squeeze(self.u[:, :, iz])

    def to_Scalar_field_XZ(self,
                           iy0=None,
                           y0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Scalar_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Parameters:
            iy0 (int): position i of y data in array
            y0 (float): position y to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        TODO:
            Simplify and change variable name clase
        """
        if is_class is True:
            field_output = Scalar_field_XZ(x=self.x,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if iy0 is None:
                iy, tmp1, tmp2 = nearest(self.y, y0)
            else:
                iy = iy0
            field_output.u = np.squeeze(self.u[:, iy, :])
            return field_output

        if matrix is True:
            if iy0 is None:
                iy, tmp1, tmp2 = nearest(self.y, y0)
            else:
                iy = iy0
            return np.squeeze(self.u[:, iy, :])

    def to_Scalar_field_YZ(self,
                           ix0=None,
                           x0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Scalar_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Parameters:
            ix0 (int): position i of x data in array
            x0 (float): position x to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        TODO:
            Simplify and change variable name clase
        """
        if is_class is True:
            field_output = Scalar_field_XZ(x=self.y,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if ix0 is None:
                ix, tmp1, tmp2 = nearest(self.x, x0)
            else:
                iy = iy0
            field_output.u = np.squeeze(self.u[ix, :, :])
            return field_output

        if matrix is True:
            if ix0 is None:
                ix, tmp1, tmp2 = nearest(self.x, x0)
            else:
                ix = ix0
            return np.squeeze(self.u[ix, :, :])

    def to_Z(self,
             kind='amplitude',
             x0=None,
             y0=None,
             has_draw=True,
             z_scale='mm'):
        """pass results to u(z). Only one of the first two variables (iy0,y0) and (ix0,x0) should be used.

        Parameters:
            kind (str): 'amplitude', 'intensity', 'phase'
            x0 (float): position x to extract
            y0 (float): position y to extract
            has_draw (bool): draw the field
            z_scale (str): 'mm', 'um'

        Returns:
            z (numpy.array): array with z
            field (numpy.array): amplitude, intensity or phase of the field
        """
        ix, _, _ = nearest(self.y, y0)
        iy, _, _ = nearest(self.x, x0)

        u = np.squeeze(self.u[ix, iy, :])

        if kind == 'amplitude':
            y = np.abs(u)
        elif kind == 'intensity':
            y = np.abs(u)**2
        elif kind == 'phase':
            y = np.angle(u)

        if has_draw is True:
            if z_scale == 'mm':
                plt.plot(self.z / mm, y, 'k', lw=2)
                plt.xlabel('$z\,(mm)$')
                plt.xlim(left=self.z[0] / mm, right=self.z[-1] / mm)

            elif z_scale == 'um':
                plt.plot(self.z, y, 'k', lw=2)
                plt.xlabel('$z\,(\mu m)$')
                plt.xlim(left=self.z[0], right=self.z[-1])

            plt.ylabel(kind)

        return self.z, y

    def beam_widths(self,
                    kind='FWHM2D',
                    has_draw=[True, False],
                    percentaje=0.5,
                    verbose=False):
        """Determines the widths of the beam

        Parameters:
            kind (str): kind of algorithm: 'sigma4', 'FWHM2D'
            has_draw (bool, bool): First for complete analysis, second for all FWHM2D computations
            verbose (bool): prints info
        Returns:
            beam_width_x (np.array)
            beam_width_y (np.array)
            principal_axis_z (np.array)
        """

        # u_zi = Scalar_field_XY(self.x, self.y, self.wavelength)
        beam_width_x = np.zeros_like(self.z)
        beam_width_y = np.zeros_like(self.z)
        principal_axis_z = np.zeros_like(self.z)
        beam_width = np.zeros_like(self.z)

        for i, zi in enumerate(self.z):
            u_prop_mat = self.u[:, :, i].squeeze()

            if kind == 'sigma4':
                dx, dy, principal_axis, moments = beam_width_2D(
                    self.x, self.y,
                    np.abs(u_prop_mat)**2)

            elif kind == 'FWHM2D':
                intensity = np.abs(u_prop_mat)**2
                # intensity = correlate2d(
                #     intensity, np.ones((3, 3)) / 9, mode='same')
                dx, dy = FWHM2D(self.x,
                                self.y,
                                intensity,
                                percentaje=percentaje,
                                remove_background='min',
                                has_draw=has_draw[1])
                principal_axis = 0.

            beam_width_x[i] = dx
            beam_width_y[i] = dy
            principal_axis_z[i] = principal_axis

            if verbose is True:
                print("{:2.0f} ".format(i))

        beam_width = np.sqrt(beam_width_x**2 + beam_width_y**2)
        if has_draw[0] is True:
            plt.figure()
            plt.plot(self.z, beam_width, 'r', label='axis')
            plt.xlabel("z ($\mu$m)")
            plt.ylabel("widths ($\mu$m)")
            plt.legend()

        return beam_width_x, beam_width_y, principal_axis_z

    def surface_detection(self, mode=1, min_incr=0.01, has_draw=False):
        """detect edges of variation in refraction index

        Parameters:
            min_incr (float): minimum incremental variation to detect.
            has_draw (bool): if True, it draws the surface.        """

        if mode == 0:
            diff1 = gradient(np.abs(self.n), axis=0)
            diff2 = gradient(np.abs(self.n), axis=1)
            diff3 = gradient(np.abs(self.n), axis=2)
        elif mode == 1:
            diff1 = diff(np.abs(self.n), axis=0)
            diff2 = diff(np.abs(self.n), axis=1)
            diff3 = diff(np.abs(self.n), axis=2)
            diff1 = np.append(diff1, np.zeros((len(self.x), 1, 1)), axis=0)
            diff2 = np.append(diff2, np.zeros((1, len(self.y), 1)), axis=1)
            diff2 = np.append(diff2, np.zeros(1, 1, len(self.z)), axis=2)

        t = np.abs(diff1) + np.abs(diff2) + np.abs(diff3)

        ix, iy, iz = (t > min_incr).nonzero()

        if has_draw is True:
            plt.figure()
            extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]
            plt.imshow(t, extent=extension, alpha=0.5)

        return self.x[ix], self.x[iy], self.z[iz]

    def draw_proposal(self,
                      kind='intensity',
                      logarithm=0,
                      normalize='maximum',
                      draw_borders=False,
                      filename='',
                      scale='',
                      min_incr=0.0005,
                      reduce_matrix='standard',
                      colorbar_kind=False,
                      colormap_kind="gist_heat"):
        """Draws  XYZ field.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', 'real'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            draw_borders (bool): If True draw edges of objects
            filename (str): if not '' stores drawing in file,
            scale (str): '', 'scaled', 'equal', scales the XY drawing
            min_incr: incrimum increment in refraction index for detecting edges
            reduce_matrix (int, int), 'standard' or False: when matrix is enormous, we can reduce it only for drawing purposes. If True, reduction factor
        """

        if reduce_matrix is False:
            amplitude, intensity, phase = field_parameters(self.u)
        elif reduce_matrix == 'standard':
            num_x = len(self.x)
            num_y = len(self.y)
            num_z = len(self.z)
            reduction_x = int(num_x / 2000)
            reduction_y = int(num_y / 2000)
            reduction_z = int(num_z / 2000)

            if reduction_x == 0:
                reduction_x = 1
            if reduction_y == 0:
                reduction_y = 1
            if reduction_z == 0:
                reduction_z = 1
            u_new = self.u[::reduction_x, ::reduction_y, ::reduction_z]
            amplitude, intensity, phase = field_parameters(u_new)
        else:
            u_new = self.u[::reduce_matrix[0], ::reduce_matrix[1], ::
                           reduce_matrix[2]]
            amplitude, intensity, phase = field_parameters(u_new)

        extension = [
            self.z[0], self.z[-1], self.x[0], self.x[-1], self.y[0], self.y[-1]
        ]

        plt.figure()

        if kind == 'intensity':
            I_drawing = intensity
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'amplitude':
            I_drawing = amplitude
            I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        elif kind == 'phase':
            I_drawing = phase
        elif kind == 'real':
            I_drawing = np.real(self.u)
        else:
            print("bad kind parameter")
            return

        h1 = plt.imshow(I_drawing,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=extension)
        plt.xlabel('z ($\mu m$)')
        plt.ylabel('x ($\mu m$)')
        plt.zlabel('y ($\mu m$)')

        plt.axis(extension)
        if colorbar_kind not in (False, '', None):
            plt.colorbar(orientation=colorbar_kind)

        h1.set_cmap(colormap_kind)  # OrRd # Reds_r gist_heat
        plt.clim(I_drawing.min(), I_drawing.max())

        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            if self.borders is None:
                self.surface_detection(1, min_incr, reduce_matrix)
            plt.plot(self.borders[0], self.borders[1], 'w.', ms=1)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_XY(self,
                z0=5 * mm,
                kind='intensity',
                logarithm=0,
                normalize='maximum',
                title='',
                filename='',
                cut_value='',
                has_colorbar='False',
                reduce_matrix=''):
        """ longitudinal profile XY at a given z value

        Parameters:
            z0 (float): value of z for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'contour'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            filename (str): if not '' stores drawing in file,
            cut_value (float): if provided, maximum value to show
            has_colorbar (bool): if True draws the colorbar
            reduce_matrix ()
        """

        ufield = self.to_Scalar_field_XY(iz0=None,
                                         z0=z0,
                                         is_class=True,
                                         matrix=False)
        ufield.draw(kind=kind,
                    logarithm=logarithm,
                    normalize=normalize,
                    title=title,
                    filename=filename,
                    cut_value=cut_value,
                    has_colorbar=has_colorbar,
                    reduce_matrix=reduce_matrix)

    def draw_XZ(self,
                y0=0 * mm,
                logarithm=0,
                normalize='',
                draw_borders=False,
                filename=''):
        """Longitudinal profile XZ at a given x0 value.

        Parameters:
            y0 (float): value of y for interpolation
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Scalar_field_XZ(y0=y0)
        intensity = np.abs(ufield.u)**2

        if logarithm == 1:
            intensity = np.log(intensity + 1)

        if normalize == 'maximum':
            intensity = intensity / intensity.max()
        if normalize == 'area':
            area = (self.x[-1] - self.x[0]) * (self.z[-1] - self.z[0])
            intensity = intensity / area
        if normalize == 'intensity':
            intensity = intensity / (intensity.sum() / len(intensity))

        h1 = plt.imshow(intensity,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=[
                            self.z[0] / 1000, self.z[-1] / 1000, self.y[0],
                            self.y[-1]
                        ])
        plt.xlabel('z (mm)', fontsize=16)
        plt.ylabel('x $(um)$', fontsize=16)
        plt.title('intensity XZ', fontsize=20)
        h1.set_cmap(
            self.params_drawing['color_intensity'])  # OrRd # Reds_r gist_heat
        plt.colorbar()

        # -----------------     no functiona de momento -----------------
        if draw_borders is True:
            x_surface, y_surface, z_surface, x_draw_intensity, y_draw_intensity, z_draw_intensity = self.surface_detection(
            )
            plt.plot(y_draw_intensity, z_draw_intensity, 'w.', ms=2)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_YZ(self,
                x0=0 * mm,
                logarithm=0,
                normalize='',
                draw_borders=False,
                filename=''):
        """Longitudinal profile YZ at a given x0 value.

        Parameters:
            x0 (float): value of x for interpolation
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Scalar_field_YZ(x0=x0)
        intensity = np.abs(ufield.u)**2

        if logarithm == 1:
            intensity = np.log(intensity + 1)

        if normalize == 'maximum':
            intensity = intensity / intensity.max()
        if normalize == 'area':
            area = (self.y[-1] - self.y[0]) * (self.z[-1] - self.z[0])
            intensity = intensity / area
        if normalize == 'intensity':
            intensity = intensity / (intensity.sum() / len(intensity))

        h1 = plt.imshow(intensity,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=[
                            self.z[0] / 1000, self.z[-1] / 1000, self.y[0],
                            self.y[-1]
                        ])
        plt.xlabel('z (mm)', fontsize=16)
        plt.ylabel('y $(um)$', fontsize=16)
        plt.title('intensity YZ', fontsize=20)
        h1.set_cmap(
            self.params_drawing['color_intensity'])  # OrRd # Reds_r gist_heat
        plt.colorbar()

        # -----------------     no functiona de momento -----------------
        if draw_borders is True:
            x_surface, y_surface, z_surface, x_draw_intensity, y_draw_intensity, z_draw_intensity = self.surface_detection(
            )
            plt.plot(y_draw_intensity, z_draw_intensity, 'w.', ms=2)

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_XYZ(self,
                 kind='intensity',
                 logarithm=False,
                 normalize='',
                 pixel_size=(128, 128, 128)):
        """Draws  XZ field.

        Parameters:
            kind (str): type of drawing: 'intensity', 'phase', 'real_field'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            pixel_size (float, float, float): pixels for drawing
            """
        try:
            from .utils_slicer import slicerLM
            is_slicer = True
        except ImportError:
            print("slicerLM is not loaded.")
            is_slicer = False

        try:
            from mayavi import mlab
            is_mayavi = True
        except ImportError:
            print("mayavi.mlab is not imported.")
            is_mayavi = False

        if is_slicer:
            u_xyz_r = self.cut_resample(num_points=(128, 128, 128),
                                        new_field=True)

            if kind == 'intensity' or kind == '':
                drawing = np.abs(u_xyz_r.u)**2
            if kind == 'phase':
                drawing = np.angle(u_xyz_r.u)
            if kind == 'real_field':
                drawing = np.real(u_xyz_r.u)

            if logarithm == 1:
                drawing = np.log(drawing**0.5 + 1)

            if normalize == 'maximum':
                factor = max(0, drawing.max())
                drawing = drawing / factor

            slicerLM(drawing)
        else:
            return

    def draw_volume(self, logarithm=0, normalize='', maxintensity=None):
        """Draws  XYZ field with mlab

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            maxintensity (float): maximum value of intensity

        TODO:
            Simplify, drawing
            include kind and other parameters of draw
        """

        try:
            from mayavi import mlab
            is_mayavi = True
        except ImportError:
            print("mayavi.mlab is not imported.")
            is_mayavi = False

        if is_mayavi:

            intensity = np.abs(self.u)**2

            if logarithm == 1:
                intensity = np.log(intensity + 1)

            if normalize == 'maximum':
                intensity = intensity / intensity.max()
            if normalize == 'area':
                area = (self.y[-1] - self.y[0]) * (self.z[-1] - self.z[0])
                intensity = intensity / area
            if normalize == 'intensity':
                intensity = intensity / (intensity.sum() / len(intensity))

            if maxintensity is None:
                intMin = intensity.min()
                intMax = intensity.max()
            else:
                intMin = maxintensity[0]
                intMax = maxintensity[1]

            mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
            mlab.clf()
            source = mlab.pipeline.scalar_field(intensity)
            mlab.pipeline.volume(source,
                                 vmin=intMin + 0.1 * (intMax - intMin),
                                 vmax=intMin + 0.9 * (intMax - intMin))
            # mlab.view(azimuth=185, elevation=0, distance='auto')
            print("Close the window to continue.")
            mlab.show()
        else:
            return

    def draw_refraction_index(self, kind='real'):
        """Draws XYZ refraction index with slicer

        Parameters:
            kind (str): 'real', 'imag', 'abs'
        """
        try:
            from .utils_slicer import slicerLM
            is_slicer = True
        except ImportError:
            print("slicerLM is not loaded.")
            is_slicer = False

        try:
            from mayavi import mlab
            is_mayavi = True
        except ImportError:
            print("mayavi.mlab is not imported.")
            is_mayavi = False

        if is_slicer:

            print("close the window to continue")
            if kind == 'real':
                slicerLM(np.real(self.n))
            elif kind == 'imag':
                slicerLM(np.imag(self.n))
            elif kind == 'abs':
                slicerLM(np.abs(self.n))
        else:
            return

    def video(self,
              filename='',
              kind='intensity',
              fps=15,
              frame=True,
              axis='auto',
              verbose=False,
              directory_name='new_video'):
        """Makes a video in the range given by self.z.

        Parameters:
            filename (str): filename (.avi, .mp4)
            kind (str): type of drawing:  'intensity', 'phase'.
            fps (int): frames per second
            frame (bool): figure with or without axis.
            verbose (bool): If True prints

        TODO:
            Implement kind, now only intensity
            include logarithm and normalize
            check
        """
        def f(x, kind):
            # return x
            amplitude, intensity, phase = field_parameters(
                x, has_amplitude_sign=True)
            if kind == 'intensity':
                # np.log(1 * intensity + 1)
                return intensity
            elif kind == 'phase':
                return phase

            elif kind == 'real':
                return np.real(x)

            else:
                return "no correct kind in video"

        if kind == 'intensity':
            cmap1 = self.params_drawing['color_intensity']
        elif kind == 'phase':
            cmap1 = self.params_drawing['color_phase']

        elif kind == 'real':
            cmap1 = self.params_drawing['color_real']

        else:
            return "no correct kind in video"

        file, extension = os.path.splitext(filename)
        if extension == '.avi':
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=fps)  # codec='ffv1'
        elif extension == '.mp4':
            Writer = anim.writers['ffmpeg']
            writer = Writer(fps=fps, bitrate=1e6,
                            codec='mpeg4')  # codec='mpeg4',
        else:
            print("file needs to be .avi or .mp4")
            print("No writer is available. is .png? Then correct.")
        xmin, xmax, ymin, ymax = self.x[0], self.x[-1], self.y[0], self.y[-1]

        if frame is True:
            plt.ion()
            fig, axes = plt.subplots(nrows=1)
            ax = plt.gca()
            plt.axis(axis)
        else:
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])

        frame = self.to_Scalar_field_XY(iz0=0,
                                        z0=None,
                                        is_class=True,
                                        matrix=False)

        intensity_global = f(self.u, kind).max()
        intensity = f(frame.u, kind)

        image = ax.imshow(intensity,
                          interpolation='bilinear',
                          aspect='equal',
                          origin='lower',
                          extent=[xmin, xmax, ymin, ymax])
        image.set_cmap(cmap1)  # seismic coolwarm gist_heat
        fig.canvas.draw()

        n_frames = len(self.z)

        if extension == '.png':
            current_directory = os.getcwd()
            try:
                os.makedirs(directory_name)
                print("new directory: {}".format(directory_name))
            except OSError:
                print("this directory is not new: overwrite.")
                pass

            os.chdir(directory_name)

            for i_prog in range(n_frames):
                frame = self.to_Scalar_field_XY(iz0=i_prog,
                                                z0=None,
                                                is_class=True,
                                                matrix=False)
                intensity = f(frame.u, kind)
                image.set_array(intensity)
                if kind == 'intensity':
                    image.set_clim(vmax=intensity_global)
                elif kind == 'phase':
                    image.set_clim(-np.pi, np.pi)
                texto = "z = {:2.3f} mm".format(self.z[i_prog] / mm)
                plt.xlabel("x (microns)")
                plt.ylabel("y (microns)")
                plt.title(texto)
                plt.draw()
                plt.savefig("{}_{:04.0f}.png".format(file, i_prog))
                if verbose:
                    if (sys.version_info > (3, 0)):
                        print(
                            "{} de {}: z={}, max= {:2.2f} min={:2.2f}".format(
                                i_prog, n_frames, self.z[i_prog] / mm,
                                intensity.max(), intensity.min()),
                            end='\r')
                    else:
                        print((
                            "{} de {}: z={}, max= {:2.2f} min={:2.2f}").format(
                                i_prog, n_frames, self.z[i_prog] / mm,
                                intensity.max(), intensity.min()))
            os.chdir(current_directory)

        elif extension == '.avi' or extension == '.mp4':
            with writer.saving(fig, filename, 300):
                for i_prog in range(n_frames):
                    frame = self.to_Scalar_field_XY(iz0=i_prog,
                                                    z0=None,
                                                    is_class=True,
                                                    matrix=False)
                    intensity = f(frame.u, kind)
                    image.set_array(intensity)
                    if kind == 'intensity':
                        image.set_clim(vmax=intensity_global)
                    elif kind == 'phase':
                        image.set_clim(-np.pi, np.pi)
                    texto = "z = {:2.3f} mm".format(self.z[i_prog] / mm)
                    plt.xlabel("x (microns)")
                    plt.ylabel("y (microns)")
                    plt.title(texto)
                    plt.draw()
                    writer.grab_frame(facecolor='k')
                    if verbose:
                        if (sys.version_info > (3, 0)):
                            print("{} de {}: z={}, max= {:2.2f} min={:2.2f}".
                                  format(i_prog, n_frames, self.z[i_prog] / mm,
                                         intensity.max(), intensity.min()),
                                  end='\r')
                        else:
                            print(("{} de {}: z={}, max= {:2.2f} min={:2.2f}"
                                   ).format(i_prog,
                                            n_frames, self.z[i_prog] / mm,
                                            intensity.max(), intensity.min()))

        plt.close()

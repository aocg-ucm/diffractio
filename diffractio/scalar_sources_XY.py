# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_source_XY class for defining sources.
Its parent is Scalar_field_XY.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.u - field XY
    * self.wavelength - wavelength of the incident field. The field is monocromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * plane_wave
    * gauss_beam
    * spherical_wave
    * vortex_beam
    * laguerre_beam
    * hermite_gauss_beam
    * zernike_beam
    * bessel_beam
    * plane_waves_dict
    * plane_waves_several_inclined
    * gauss_beams_several_parallel
    * gauss_beams_several_inclined

*Also*
    * laguerre_polynomial_nk
    * fZernike
    * delta_kronecker
"""

from math import factorial

from numpy import arctan2, cos, exp, pi, sign, sin, sqrt, zeros
from scipy.special import j0, j1, jv
from scipy.special.orthogonal import hermite

from diffractio import degrees, np, um
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.utils_math import fZernike, laguerre_polynomial_nk


class Scalar_source_XY(Scalar_field_XY):
    """Class for XY scalar sources.

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
        super(self.__class__, self).__init__(x, y, wavelength, info)
        self.type = 'Scalar_source_XY'

    # ______________________________________________________________________
    # pasamos a definir fuentes

    def plane_wave(self, A=1, theta=0 * degrees, phi=0 * degrees, z0=0 * um):
        """Plane wave. self.u = A * exp(1.j * k * (self.x * sin(theta) + z0 * cos(theta)))

        Parameters:
            A (float): maximum amplitude
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift
        """
        k = 2 * pi / self.wavelength
        self.u = A * exp(1.j * k * (
            self.X * sin(theta) * sin(phi) + self.Y * cos(theta) * sin(phi)
        ) + z0 * cos(phi))

    def gauss_beam(self,
                   r0,
                   w0,
                   z0=0 * um,
                   A=1,
                   theta=0. * degrees,
                   phi=0 * degrees):
        """Gauss Beam.

        Parameters:
            r0 (float, float): (x,y) position of center
            w0 (float, float): (wx,wy) minimum beam width
            z0 (float): position of beam width
            A (float): maximum amplitude
            theta (float): angle in radians (direction of propagation)
            phi (float): angle in radians (direction of propagation)

        Todo:
            generalize definition
        """

        if isinstance(w0, (float, int, complex)):
            w0 = (w0, w0)

        w0x, w0y = w0
        w0 = sqrt(w0x * w0y)
        x0, y0 = r0
        k = 2 * pi / self.wavelength

        # only for x axis.
        z_rayleigh = k * w0x**2 / 2

        phaseGouy = arctan2(z0, z_rayleigh)

        wx = w0x * sqrt(1 + (z0 / z_rayleigh)**2)
        wy = w0y * sqrt(1 + (z0 / z_rayleigh)**2)
        w = sqrt(wx * wy)

        if z0 == 0:
            R = 1e10
        else:
            R = z0 * (1 + (z_rayleigh / z0)**2)

        amplitude = A * w0 / w * exp(-(self.X - x0)**2 / (wx**2) -
                                     (self.Y - y0)**2 / (wy**2))
        phase1 = exp(
            1.j * k *
            (self.X * sin(theta) * sin(phi) + self.Y * cos(theta) * sin(phi)
             ))  # rotation
        phase2 = exp(1j * (k * z0 - phaseGouy + k * (self.X**2 + self.Y**2) /
                           (2 * R)))

        self.u = amplitude * phase1 * phase2

    def spherical_wave(self,
                       A=1,
                       r0=(0 * um, 0 * um),
                       z0=-1000 * um,
                       mask=True,
                       radius=100 * um,
                       normalize=False):
        """Spherical wave.

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): (x,y) position of source
            z0 (float): z position of source
            mask (bool): If true, masks the spherical wave with radius
            radius (float): size of slit for mask
            normalize (bool): If True, maximum of field is 1
        """

        k = 2 * pi / self.wavelength
        x0, y0 = r0

        # Centrado radius de la mask y distance al origen emisor
        R2 = (self.X - x0)**2 + (self.Y - y0)**2
        Rz = sqrt((self.X - x0)**2 + (self.Y - y0)**2 + z0**2)

        # Definicion de la mask circular
        if mask is True:
            amplitude = (R2 <= radius**2)
        else:
            amplitude = 1

        # Onda esferica
        self.u = amplitude * A * exp(-1.j * sign(z0) * k * Rz) / Rz

        if normalize is True:
            self.u = self.u / np.abs(self.u.max() + 1.012034e-12)

    def vortex_beam(self, r0, w0, m):
        """Vortex beam.

        Parameters:
            r0 (float, float): (x,y) position of source
            w0 (float): width of the vortex beam
            m (int): order of the vortex beam

        Example:
            vortex_beam(r0=(0 * um, 0 * um), w0=100 * um, m=1)
        """

        x0, y0 = r0
        # Definicion del vortice
        amplitude = ((self.X - x0) + 1.j * sign(m) *
                     (self.Y - y0))**np.abs(m) * exp(-(
                         (self.X - x0)**2 + (self.Y - y0)**2) / w0**2)
        self.u = amplitude

    def laguerre_beam(self, r0, w0, z, p, l):
        """Laguerre beam.

        Parameters:
            r0 (float, float): (x,y) position of source
            w0 (float): width of the vortex beam
            z (float): distance
            p (int): order of the laguerre_beam
            l (int): order of the laguerre_beam

        Example:
            laguerre_beam(r0=(0 * um, 0 * um),  w0=100 * um,  z=0.01 * um,  p=0, l=0)
        """
        # Vector de onda
        k = 2 * pi / self.wavelength
        # rango de Rayleigh
        zR = pi * w0**2 / self.wavelength

        # Definicion de parametros
        Rz = z * sqrt(1 + (zR / z)**2)
        wz = w0 * sqrt(1 + (z / zR)**2)
        f_gouy = k * z - arctan2(z, zR)

        R2 = (self.X - r0[0])**2 + (self.Y - r0[1])**2
        THETA = arctan2(self.X, self.Y)

        # Definicion de los terminos producto
        t1 = exp(-1.j * k * R2 / (2 * Rz) - R2 / wz**2 + 1.j *
                 (2 * p + l + 1) * f_gouy)
        t2 = exp(-1.j * l * THETA)
        t3 = ((-1)**p) * (R2 / wz**2)**(l / 2)
        t4 = laguerre_polynomial_nk(2 * R2 / wz**2, p, l)
        # El field es el producto t1*t2*t3*t4
        self.u = t1 * t2 * t3 * t4

    def hermite_gauss_beam(self, A, r0, w0, m, n, c_mn):
        """Hermite Gauss beam.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam
            r0 (float, float): (x,y) position of source
            w0 (float): width of the beam
            m (list): list of integers with orders
            n (list): list of integers with orders
            c_mn (list): list of integers with coefficients

        Example:
             hermite_gauss_beam(A=1, r0, w0=100 * um, m=[1, 3, 3, 5, 5, 5], n=[1, 1, 3, 1, 3, 5], c_mn=[.25, 1, 1, 1, 1, 1])
        """
        x0, y0 = r0
        intesity = zeros(self.X.shape, dtype=np.float)

        for s in range(len(m)):
            Ix = (hermite(m[s])(sqrt(2 * pi) * (
                self.X - x0) / w0) * exp(-pi * (self.X - x0)**2 / w0**2))**2
            Iy = (hermite(n[s])(sqrt(2 * pi) * (
                self.Y - y0) / w0) * exp(-pi * (self.Y - y0)**2 / w0**2))**2
            f = sqrt(2) / (w0 * sqrt(2**m[s] * factorial(m[s])
                                     ) * sqrt(2**n[s] * factorial(n[s])))

            intesity = intesity + f * c_mn[s] * Ix * Iy

        self.u = A * intesity

    def zernike_beam(self, A, r0, radius, n, m, c_nm, mask=True):
        """Zernike beam.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam
            r0 (float, float): (x,y) position of source
            radius (float): width of the beam
            n (list): list of integers with orders
            m (list): list of integers with orders
            c_nm (list): list of integers with coefficients
            mask (bool): if True a mask of radius is provided

        Example:
             zernike_beam(A=1, r0=(0,0), radius=5 * mm, n=[1, 3, 3, 5, 5, 5], m=[1, 1, 3, 1, 3, 5], c_nm=[.25, 1, 1, 1, 1, 1], mask=True)
        """

        # normalizing for radius 1
        x0, y0 = r0
        R = sqrt((self.X - x0)**2 + (self.Y - y0)**2) / radius

        # phase as sum of Zernike functions
        phase = zeros(self.X.shape, dtype=np.float)
        for s in range(len(n)):
            phase = phase + c_nm[s] * fZernike(self.X - x0, self.Y - y0, n[s],
                                               m[s], radius)

        if mask is True:
            amplitude = (R < 1)
        else:
            amplitude = 1

        self.u = A * amplitude * exp(1.j * np.real(phase))

    def bessel_beam(self,
                    A,
                    r0,
                    alpha,
                    n=0,
                    theta=0 * degrees,
                    phi=0 * degrees,
                    z0=0):
        """Bessel beam produced by an axicon. Bessel-beams are generated using 2D axicons.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam
            r0 (float, float): (x,y) position of source
            alpha (float): angle of the beam generator
            n (int): order of the beam
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift

        References:
            F. Courvoisier, et al. "Surface nanoprocessing with nondiffracting femtosecond Bessel beams" Optics Letters Vol. 34, No. 20 3163 (2009)
        """

        x0, y0 = r0
        R = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        k = 2 * np.pi / self.wavelength
        beta = k * np.cos(alpha)

        if n == 0:
            jbessel = j0(k * np.sin(alpha) * R)
        elif n == 1:
            jbessel = j1(k * np.sin(alpha) * R)
        else:
            jbessel = jv(n, k * np.sin(alpha) * R)

        self.u = A * jbessel * np.exp(
            1j * beta * z0) * np.exp(1.j * k * (
                self.X * sin(theta) * sin(phi) + self.Y * cos(theta) * sin(phi)
            ) + z0 * cos(phi))

    def plane_waves_dict(self, params):
        """Several plane waves with parameters defined in dictionary

        Parameters:
            params: list with a dictionary:
                A (float): maximum amplitude
                theta (float): angle in radians
                phi (float): angle in radians
                z0 (float): constant value for phase shift
        """
        # Definicion del vector de onda
        k = 2 * pi / self.wavelength

        self.u = np.zeros_like(self.u, dtype=complex)
        for p in params:
            self.u = self.u + p['A'] * exp(1.j * k * (
                self.X * sin(p['theta']) * sin(p['phi']) + self.
                Y * cos(p['theta']) * sin(p['phi']) + p['z0'] * cos(p['phi'])))

    def plane_waves_several_inclined(self, A, num_beams, max_angle, z0=0):
        """Several paralel plane waves

        Parameters:
                A (float): maximum amplitude
                num_beams (int, int): number of beams in the x and y directions
                max_angle (float, float): maximum angle of the beams
                z0 (float): position of the beams
        """

        num_beams_x, num_beams_y = num_beams
        max_angle_x, max_angle_y = max_angle
        t = np.zeros_like(self.u, dtype=complex)
        anglex = max_angle_x / num_beams_x
        angley = max_angle_y / num_beams_y
        for i in range(num_beams_x):
            for j in range(num_beams_y):
                theta = np.pi / 2 - max_angle_x / 2 + anglex * (i + 0.5)
                phi = np.pi / 2 - max_angle_y / 2 + angley * (j + 0.5)
                self.plane_wave(A, theta, phi, z0)
                t = t + self.u
        self.u = t

    def gauss_beams_several_parallel(self,
                                     A,
                                     num_beams,
                                     w0,
                                     z0,
                                     r0=(0 * um, 0 * um),
                                     r_range=100 * um,
                                     theta=0 * degrees,
                                     phi=0 * degrees):
        """Several parallel gauss beams

        Parameters:
            A (float): maximum amplitude
            num_beams (int, int): number of gaussian beams (equidistintan) in x and y direction.
            w0 (float): beam width of the bemas
            z0 (float): constant value for phase shift
            r0 (float, float): central position of rays (x_c, y_c)
            r_range (float, float): range of rays x, y
            theta (float): angle
            phi (float): angle
        """

        x_range, y_range = r_range
        x_central, y_central = r0
        num_beams_x, num_beams_y = num_beams
        t = np.zeros_like(self.u, dtype=complex)
        dist_x = x_range / num_beams_x
        dist_y = y_range / num_beams_y
        for i in range(num_beams_x):
            xi = x_central - x_range / 2 + dist_x * (i + 0.5)
            for j in range(num_beams_y):
                yi = y_central - y_range / 2 + dist_y * (j + 0.5)
                self.gauss_beam(
                    r0=(xi, yi), w0=w0, z0=z0, A=A, theta=theta, phi=phi)
                t = t + self.u
        self.u = t

    def gauss_beams_several_inclined(self, A, num_beams, w0, r0, z0,
                                     max_angle):
        """Several inclined gauss beams

        Parameters:
            A (float): maximum amplitude
            num_beams (int, int): number of gaussian beams (equidistintan) in x and y direction.
            w0 (float): beam width
            r0 (float, float): central position of rays (x_c, y_c)
            z0 (float): constant value for phase shift
            max_angle (float, float): maximum angles
        """

        num_beams_x, num_beams_y = num_beams
        max_angle_x, max_angle_y = max_angle
        t = np.zeros_like(self.u, dtype=complex)
        angle_x = max_angle_x / num_beams_x
        angle_y = max_angle_y / num_beams_y
        for i in range(num_beams_x):
            thetai = np.pi / 2 - max_angle_x / 2 + angle_x * (i + 0.5)
            for j in range(num_beams_y):
                phii = np.pi / 2 - max_angle_y / 2 + angle_y * (j + 0.5)
                self.gauss_beam(
                    r0=r0, w0=w0, z0=z0, A=A, theta=thetai, phi=phii)
                t = t + self.u

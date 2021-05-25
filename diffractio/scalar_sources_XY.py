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
from scipy.special import eval_hermite, j0, j1, jv

from . import degrees, np, um
from .scalar_fields_XY import Scalar_field_XY
from .utils_math import fZernike, laguerre_polynomial_nk

# from scipy.special.orthogonal import hermite


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

    def plane_wave(self, A=1, theta=0 * degrees, phi=0 * degrees, z0=0 * um):
        """Plane wave. self.u = A * exp(1.j * k * (self.x * sin(theta) + z0 * cos(theta)))

        Parameters:
            A (float): maximum amplitude
            theta (float): angle in radians
            phi (float): angle in radians
            z0 (float): constant value for phase shift
        """
        k = 2 * pi / self.wavelength
        self.u = A * exp(1.j * k *
                         (self.X * sin(theta) * sin(phi) +
                          self.Y * cos(theta) * sin(phi)) + z0 * cos(phi))

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
            R = - z0 * (1 + (z_rayleigh / z0)**2)

        amplitude = A * w0 / w * exp(-(self.X - x0)**2 / (wx**2) -
                                     (self.Y - y0)**2 / (wy**2))
        phase1 = exp(1.j * k * (self.X * sin(theta) * sin(phi) +
                                self.Y * cos(theta) * sin(phi)))
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

    def vortex_beam(self, A, r0, w0, m):
        """Vortex beam.

        Parameters:
            A (float): Amplitude
            r0 (float, float): (x,y) position of source
            w0 (float): width of the vortex beam
            m (int): order of the vortex beam

        Example:
            vortex_beam(r0=(0 * um, 0 * um), w0=100 * um, m=1)
        """

        if isinstance(w0, (float, int, complex)):
            w0x, w0y = w0, w0
        else:
            w0x, w0y = w0

        x0, y0 = r0
        amplitude = ((self.X - x0) + 1.j * sign(m) *
                     (self.Y - y0))**np.abs(m) * np.exp(-(
                         (self.X - x0)**2 / w0x**2 + (self.Y - y0)**2 / w0y**2))

        self.u = A * amplitude / np.abs(amplitude).max()

    def hermite_gauss_beam(self, r0,
                           A,
                           w0,
                           n,
                           m,
                           z,
                           z0):
        """Hermite Gauss beam.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam.
            r0 (float, float): (x,y) position of the beam center.
            w0 (float, float): Gaussian waist.
            n (int): order in x.
            m (int): order in y.
            z (float): Propagation distance.
            z0 (float, float): Beam waist position at each dimension

        Example:
             hermite_gauss_beam(A=1, r0=(0,0), w0=(100*um, 50*um), n=2, m=3, z=0)
        """
        # Prepare space
        # E = zeros(self.X.shape, dtype=np.float)
        X = self.X - r0[0]
        Y = self.Y - r0[1]

        # Parameters
        r2 = sqrt(2)

        if isinstance(w0, (float, int, complex)):
            w0x, w0y = w0, w0
        else:
            w0x, w0y = w0

        if isinstance(z0, (float, int, complex)):
            z0x, z0y = z0, z0
        else:
            z0x, z0y = z0

        k = 2 * pi / self.wavelength

        # Calculate propagation
        zx = z - z0x
        zRx = k * w0x**2 / 2
        wx = w0x * sqrt(1 + zx**2 / zRx**2)
        if zx == 0:
            Rx = np.inf
        else:
            Rx = zx + zRx**2 / zx

        zy = z - z0y
        zRy = k * w0y**2 / 2
        wy = w0y * sqrt(1 + zy**2 / zRy**2)
        if zy == 0:
            Ry = np.inf
        else:
            Ry = zy + zRy**2 / zy

        # Calculate amplitude
        A = A * sqrt(2**(1 - n - m) /
                     (pi * factorial(n) * factorial(m))) * sqrt(w0x * w0y /
                                                                (wx * wy))
        Ex = eval_hermite(n, r2 * X / wx) * exp(-X**2 / wx**2)
        Ey = eval_hermite(m, r2 * Y / wy) * exp(-Y**2 / wy**2)

        # Calculate phase
        Ef = exp(1j * k * (X**2 / Rx + Y**2 / Ry)) * exp(
            -1j * (0.5 + n) * np.arctan(zx / zRx)) * exp(
                -1j * (0.5 + m) * np.arctan(zy / zRy)) * exp(1j * k *
                                                             (zx + zy) / 2)

        self.u = A * Ex * Ey * Ef

    def laguerre_beam(self, r0, A, w0, n, l, z, z0):
        """Laguerre beam.

        Parameters:
            A (float): amplitude of the Hermite Gauss beam.
            r0 (float, float): (x,y) position of the beam center.
            w0 (float): Gaussian waist.
            n (int): radial order.
            l (int): angular order.
            z (float): Propagation distance.
            z0 (float): Beam waist position.

        Example:
            laguerre_beam(A=1, r0=(0 * um, 0 * um),  w0=1 * um,  p=0, l=0,  z=0)
        """
        # Prepare space
        X = self.X - r0[0]
        Y = self.Y - r0[1]
        Ro2 = X**2 + Y**2
        Ro = np.sqrt(Ro2)
        Th = np.arctan2(Y, X)

        # Parameters
        r2 = sqrt(2)
        z = z - z0
        k = 2 * pi / self.wavelength

        # Calculate propagation
        zR = k * w0**2 / 2
        w = w0 * sqrt(1 + z**2 / zR**2)
        if z == 0:
            R = np.inf
        else:
            R = z + zR**2 / z

        # Calculate amplitude
        A = A * w0 / w
        Er = laguerre_polynomial_nk(2 * Ro2 / w**2, n, l) * exp(
            -Ro2 / w**2) * (r2 * Ro / w)**l

        # Calculate phase
        Ef = exp(1j * (k * Ro2 / R + l * Th)) * \
            exp(-1j * (1 + n) * np.arctan(z / zR))

        self.u = A * Er * Ef

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
                    n,
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

        self.u = A * jbessel * np.exp(1j * beta * z0) * np.exp(
            1.j * k *
            (self.X * sin(theta) * sin(phi) + self.Y * cos(theta) * sin(phi)) +
            z0 * cos(phi))

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
            self.u = self.u + p['A'] * exp(
                1.j * k *
                (self.X * sin(p['theta']) * sin(p['phi']) + self.Y *
                 cos(p['theta']) * sin(p['phi']) + p['z0'] * cos(p['phi'])))

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
                                     r0,
                                     A,
                                     num_beams,
                                     w0,
                                     z0,
                                     r_range,
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
                self.gauss_beam(r0=(xi, yi),
                                w0=w0,
                                z0=z0,
                                A=A,
                                theta=theta,
                                phi=phi)
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
                self.gauss_beam(r0=r0,
                                w0=w0,
                                z0=z0,
                                A=A,
                                theta=thetai,
                                phi=phii)
                t = t + self.u

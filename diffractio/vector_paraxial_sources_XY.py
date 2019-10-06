# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_paraxial_source_XY class for defining sources.
Its parent is Vector_paraxial_field_XY.

The main atributes are:
    * x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
    * y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
    * wavelength (float): wavelength of the incident field
    * info (str): String with info about the simulation


The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * plane_wave
    * radial_wave
    * transversal_wave
    * gauss_wave
    * hermite_gauss_wave
    * local_polarized_vector_beam
    * local_polarized_vector_beam_radial
    * local_polarized_vector_beam_hybrid
"""

from diffractio import (degrees, epsilon_0, mm, mu_0, nm, np, plt, sp,
                        speed_of_light, um)
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_math import vector_product
from diffractio.utils_optics import normalize
from diffractio.vector_paraxial_fields_XY import Vector_paraxial_field_XY


class Vector_paraxial_source_XY(Vector_paraxial_field_XY):
    """Class for vectorial fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field

    """

    def __init__(self, x, y, wavelength, info=''):
        super(self.__class__, self).__init__(x, y, wavelength, info)

    def plane_wave(self, A=1, v=[1, 0], theta=0 * degrees, phi=0 * degrees):
        """Plane wave.

        self.Ex = v[0] * A * np.exp(1j * k * (self.X * sx + self.Y * sy))
        self.Ey = v[1] * A * np.exp(1j * k * (self.X * sx + self.Y * sy))


        Parameters:
            A (float): maximum amplitude
            v (float, float): vector of polarization (normalized, if not is normalized)
            theta (float): angle in radians
            phi (float): angle in radians
        """

        v = normalize(v)
        k = 2 * np.pi / self.wavelength
        sx = np.sin(theta) * np.cos(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.cos(theta)
        v2 = (-sx * v[0] - sy * v[1]) / sz
        v = np.append(v, v2)
        s = [sx, sy, sz]

        self.Ex = v[0] * A * np.exp(1j * k * (self.X * sx + self.Y * sy))
        self.Ey = v[1] * A * np.exp(1j * k * (self.X * sx + self.Y * sy))

    def radial_wave(self, A=1, x_center=(0 * um, 0 * um), radius=0):
        """Radial wave.

        Parameters:
            A (float): maximum amplitude
            x_center (float, float): center of radiality
            radius (float): mask for circle if radius>0.

        Todo:
            no cuadra S y vectores v complejos para los productos

            check perpendicularity between v and k vector
        """

        # normalizamos v
        vx = (self.X - x_center[0])
        vy = (self.Y - x_center[1])
        theta = np.arctan2(vy, vx)

        self.Ex = A * np.cos(theta)
        self.Ey = -A * np.sin(theta)

        if radius > 0:
            t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
            t1.circle(r0=x_center, radius=(radius, radius), angle=0 * degrees)
            self.Ex = t1.u * self.Ex
            self.Ey = t1.u * self.Ey

    def transversal_wave(self, A=1, x_center=(0 * um, 0 * um), radius=0):
        """Transversal wave.

        Parameters:
            A (float): maximum amplitude
            x_center (float, float): center of radiality
            radius (float): mask for circle if radius >0.
        """
        # normalizamos v
        vx = (self.X - x_center[0])
        vy = (self.Y - x_center[1])

        theta = np.arctan2(vy, vx)

        # Vx = vx / np.sqrt(vx**2 + vy**2)
        # Vy = vy / np.sqrt(vx**2 + vy**2)

        self.Ex = A * np.sin(theta)
        self.Ey = A * np.cos(theta)

        if radius > 0:
            t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
            t1.circle(r0=x_center, radius=(radius, radius), angle=0 * degrees)
            self.Ex = t1.u * self.Ex
            self.Ey = t1.u * self.Ey

    def gauss(self,
              A=1,
              r0=(0 * um, 0 * um),
              z=0 * um,
              w0=(100 * um, 100 * um),
              theta=0. * degrees,
              phi=0 * degrees,
              kind='polarization',
              v=[1, 0]):
        """Electromagnetic gauss beam.

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of gauss beam
            z (float): position of beam waist
            theta (float): angle in radians
            phi (float): angle in radians
            kind (str): 'polarization', 'radial', 'transversal' (polarization uses v)
            v (float, float): polarization vector when 'polarization' is chosen
        """

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        if isinstance(r0, (float, int, complex)):
            r0 = (r0[0], r0[0])
        if isinstance(w0, (float, int, complex)):
            w0 = (w0, w0)

        w0x, w0y = w0
        x0, y0 = r0
        w0 = np.sqrt(w0x * w0y)
        k = 2 * np.pi / self.wavelength

        z0 = k * w0x**2 / 2  # distance de Rayleigh\ solo para una direccion.

        phaseGouy = np.arctan2(z, z0)

        wx = w0x * np.sqrt(1 + (z / z0)**2)
        wy = w0y * np.sqrt(1 + (z / z0)**2)
        w = np.sqrt(wx * wy)
        if z == 0:
            R = 1e10
        else:
            R = z * (1 + (z0 / z)**2)

        amplitude = A * w0 / w * np.exp(-((self.X - x0) / wx)**2 - (
            (self.Y - y0) / wy)**2)

        sx = np.sin(theta) * np.cos(phi)
        sy = np.sin(theta) * np.sin(phi)
        sz = np.cos(theta)
        #  = [sx, sy, sz]
        phase1 = np.exp(
            1.j * k * ((self.X - x0) * sx + (self.Y - y0) * sy))  # rotation
        phase2 = np.exp(1j * (k * z - phaseGouy + k * (
            (self.X - x0)**2 + (self.Y - y0)**2) / (2 * R)))

        self.u = amplitude * phase1 * phase2

        if kind == 'polarization':
            self.Ex = self.u * v[0]
            self.Ey = self.u * v[1]
        elif kind == 'radial':
            self.Ex = self.u * np.cos(angle)
            self.Ey = -self.u * np.sin(angle)
        elif kind == 'transversal':
            self.Ex = self.u * np.sin(angle)
            self.Ey = self.u * np.cos(angle)

    def hermite_gauss_wave(self,
                           A=1,
                           r0=(0 * um, 0 * um),
                           w=100 * um,
                           m=[1, 3, 3, 5, 5, 5],
                           n=[1, 1, 3, 1, 3, 5],
                           c_mn=[.25, 1, 1, 1, 1, 1],
                           kind='polarization',
                           v=[1, 0]):
        """Electromagnetic hermite_gauss_wave.

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            w (float): width of beam waist
            m (list): list with components m
            n (list): list with components m
            c_mn (list): amplitude for component (m,n)
            kind (str): 'polarization', 'radial', 'transversal' (polarization uses v)
            v (float, float): polarization vector when 'polarization' is chosen
        """

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        field = Scalar_source_XY(
            x=self.x, y=self.y, wavelength=self.wavelength)
        field.hermite_gauss_beam(A, r0, w, m, n, c_mn)

        if kind == 'polarization':
            self.Ex = field.u * v[1]
            self.Ey = field.u * v[0]
        elif kind == 'radial':
            self.Ex = field.u * np.cos(angle)
            self.Ey = -field.u * np.sin(angle)
        elif kind == 'transversal':
            self.Ex = field.u * np.sin(angle)
            self.Ey = field.u * np.cos(angle)

    def local_polarized_vector_beam(self, A=1, r0=(0 * um, 0 * um), m=1,
                                    fi0=0):
        """"local radial polarized vector wave.

        References:
            Qwien Zhan 'Vectorial Optical Fields' page 33

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            m (integer): integer with order
            fi0 (float): initial phase
        """

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        delta = m * angle + fi0

        t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        t1.circle(r0=r0, radius=(self.x[-1], self.y[-1]), angle=0 * degrees)
        self.Ex = t1.u * A * np.cos(delta)
        self.Ey = t1.u * A * np.sin(delta)

    def local_polarized_vector_beam_radial(self,
                                           A=1,
                                           r0=(0 * um, 0 * um),
                                           n=1,
                                           fi0=0):
        """local radial polarized vector wave.

        References:
            Qwien Zhan 'Vectorial Optial Fields' page 36

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            m (integer): integer with order
            fi0 (float): initial phase
        """

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        r = np.sqrt(vx**2 + vy**2)
        radius = self.x[-1]
        delta = 2 * n * np.pi * r / radius + fi0

        t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        t1.circle(r0=r0, radius=(self.x[-1], self.y[-1]), angle=0 * degrees)
        self.Ex = t1.u * A * np.cos(delta)
        self.Ey = t1.u * A * np.sin(delta)

    def local_polarized_vector_beam_hybrid(self,
                                           A=1,
                                           r0=(0 * um, 0 * um),
                                           m=1,
                                           n=1,
                                           fi0=0):
        """local hibrid polarized vector wave.
            Qwien Zhan 'Vectorial Optial Fields' page 36

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            m (integer): integer with order
            n (integer): integer with order
            fi0 (float): initial phase
        """

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        r = np.sqrt(vx**2 + vy**2)
        radius = self.x[-1]
        delta = m * angle + 2 * n * np.pi * r / radius + fi0

        t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        t1.circle(r0=r0, radius=(self.x[-1], self.y[-1]), angle=0 * degrees)
        self.Ex = t1.u * A * np.cos(delta)
        self.Ey = t1.u * A * np.sin(delta)

    # def Vector_paraxial_dipole(self,
    #                            A=1,
    #                            r0=(0, 0, 0),
    #                            phase=0 * degrees,
    #                            p0=[0, 0, 1],
    #                            z=1 * mm):
    #     """Dipole, according to electric dipole approximation.
    #
    #     Parameters:
    #         A (float): maximum amplitude
    #         r0 (float, float, float): position of dipole
    #         phase (float): initial phase of dipole
    #         p0 (float): vector polarization
    #         z (float): z distance from the dipole to the plane
    #     """
    #
    #     p0 = normalize(p0)
    #     x0, y0, z0 = r0
    #
    #     Z = z * np.ones_like(self.X)
    #     S = [self.X, self.Y, Z]
    #     w = 2 * np.pi * speed_of_light / self.wavelength
    #
    #     # distance al dipolo
    #     R = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2 + (Z - z0)**2)
    #
    #     Px = p0[0] * np.ones_like(self.X)
    #     Py = p0[1] * np.ones_like(self.X)
    #     Pz = p0[2] * np.ones_like(self.X)
    #     P = [Px, Py, Pz]
    #
    #     E = A * (mu_0 * (4 * np.pi * R)) * (-w**2) * vector_product(
    #         vector_product(P, S), S) * np.exp(1j * phase)
    #
    #     self.Ex = E[0]
    #     self.Ey = E[1]

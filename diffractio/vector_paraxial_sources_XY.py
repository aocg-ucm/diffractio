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
    * local_polarized_vector_wave
    * local_polarized_vector_wave_radial
    * local_polarized_vector_wave_hybrid
"""

from diffractio import degrees, eps, np, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
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

    def constant_wave(self,
                      u=None,
                      v=(1, 0),
                      has_normalization=True,
                      radius=(0, 0)):
        """Provides a constant polarization to a scalar_source_xy

        u (Scalar_source_XY): field to apply a constant polarization.
        v (float, float): polarization vector
        normalize (bool): If True, normalize polarization vector
        """

        self = define_initial_field(self, u)

        if has_normalization:
            v = normalize(v)

        self.Ex = v[0] * self.Ex
        self.Ey = v[1] * self.Ey

        if radius[0] * radius[1] > 0:
            self.mask_circle(radius=radius)

    def radial_wave(self, u=None, r0=(0, 0), radius=(0, 0)):
        """Provides a constant polarization to a scalar_source_xy

        u (Scalar_source_XY): field to apply a constant polarization.
        r0(float, float): center of rotation
        ce (bool): If True, normalize polarization vector
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.sin(angle) * self.Ex
        self.Ey = -np.cos(angle) * self.Ey

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def transversal_wave(self, u=None, r0=(0, 0), radius=(0, 0)):
        """Provides a constant polarization to a scalar_source_xy

        u (Scalar_source_XY): field to apply a constant polarization.
        r0(float, float): center of rotation
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.cos(angle) * self.Ex
        self.Ey = np.sin(angle) * self.Ey

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def radial_inverse_wave(self, u=None, r0=(0, 0), radius=(0, 0)):
        """Provides a constant polarization to a scalar_source_xy

        u (Scalar_source_XY): field to apply a constant polarization.
        r0(float, float): center of rotation
        ce (bool): If True, normalize polarization vector
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.cos(angle) * self.Ex
        self.Ey = -np.sin(angle) * self.Ey

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def transversal_inverse_wave(self, u=None, r0=(0, 0), radius=(0, 0)):
        """Provides a constant polarization to a scalar_source_xy

        u (Scalar_source_XY): field to apply a constant polarization.
        r0(float, float): center of rotation
        ce (bool): If True, normalize polarization vector
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.sin(angle) * self.Ex
        self.Ey = np.cos(angle) * self.Ey

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def local_polarized_vector_wave(self,
                                    u,
                                    r0=(0, 0),
                                    m=1,
                                    fi0=0,
                                    radius=(0, 0)):
        """"local radial polarized vector wave.

        References:
            Qwien Zhan 'Vectorial Optical Fields' page 33

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): r0 of beam
            m (integer): integer with order
            fi0 (float): initial phase
            has_mask
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        delta = m * angle + fi0

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def local_polarized_vector_wave_radial(self,
                                           u,
                                           r0=(0 * um, 0 * um),
                                           m=1,
                                           fi0=0,
                                           radius=(0, 0)):
        """local radial polarized vector wave.

        References:
            Qwien Zhan 'Vectorial Optial Fields' page 36

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            m (integer): integer with order
            fi0 (float): initial phase
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        r = np.sqrt(vx**2 + vy**2)
        radius_0 = min(radius[0], radius[1])
        delta = 2 * m * np.pi * r / (radius_0 + eps) + fi0

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

    def local_polarized_vector_wave_hybrid(self,
                                           u,
                                           r0=(0 * um, 0 * um),
                                           m=1,
                                           n=1,
                                           fi0=0,
                                           radius=(0, 0)):
        """local hibrid polarized vector wave.
            Qwien Zhan 'Vectorial Optial Fields' page 36

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of beam
            m (integer): integer with order
            n (integer): integer with order
            fi0 (float): initial phase
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        r = np.sqrt(vx**2 + vy**2)
        radius_0 = min(radius[0], radius[1])
        delta = m * angle + 2 * n * np.pi * r / (radius_0 + eps) + fi0

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def spiral_polarized_beam(self,
                              u,
                              r0=(0 * um, 0 * um),
                              alpha=0,
                              radius=(0, 0)):
        """Define spiral polarized beams:

        Parameters:
            A (float): maximum amplitude
            r0 (float, float): center of radiality
            radius (float): mask for circle if radius >0.
            alpha (float): angle of spiral.


        Reference:
            V. Ramirez-Sanchez, G. Piquero, and M. Santarsiero,“Generation and characterization of spirally polarized fields,” J. Opt. A11,085708 (2009)
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])

        theta = np.arctan2(vy, vx)

        self.Ex = -self.Ex * np.sin(theta + alpha)
        self.Ey = self.Ey * np.cos(theta + alpha)

        if radius[0] * radius[1] > 0:
            self.mask_circle(r0=r0, radius=radius)

    def mask_circle(self, r0=(0, 0), radius=(0, 0)):
        """Mask vector field using a circular mask.

        Parameters:
            r0 (float, float): center of mask.
            radius (float, float): radius of mask
        """

        if radius in (0, None, '', []):
            radius_x = (self.x[-1] - self.x[0]) / 2
            radius_y = (self.y[-1] - self.y[0]) / 2
            radius = (radius_x, radius_y)

        if r0 in (0, None, '', []):
            r0_x = (self.x[-1] + self.x[0]) / 2
            r0_y = (self.y[-1] + self.y[0]) / 2
            r0 = (r0_x, r0_y)

        if radius[0] * radius[1] > 0:
            t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
            t1.circle(r0=r0, radius=radius, angle=0 * degrees)
            self.Ex = t1.u * self.Ex
            self.Ey = t1.u * self.Ey


def define_initial_field(EM, u):
    """Rewrites the initial field of EM in terms of u.

    EM (vector_paraxial_source_XY):
    u (scalar_source_XY, or None, or 1): if scalar_source it is written in Ex and Ey, is 1 Ex=1, Ey=1, if None, does nothing,
    """

    # check data size
    if u == 1:
        EM.Ex = np.ones_like(EM.Ex)
        EM.Ey = np.ones_like(EM.Ey)
    elif u not in ('', None, [], 0):
        EM.Ex = u.u
        EM.Ey = u.u

    return EM

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

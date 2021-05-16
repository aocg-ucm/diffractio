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
    * azimuthal_wave
    * radial_wave
    * gauss_wave
    * hermite_gauss_wave
    * local_polarized_vector_wave
    * local_polarized_vector_wave_radial
    * local_polarized_vector_wave_hybrid
"""

from py_pol.jones_vector import Jones_vector

from . import degrees, eps, np, um
from .scalar_fields_XY import Scalar_field_XY
from .scalar_masks_XY import Scalar_mask_XY
from .scalar_sources_XY import Scalar_source_XY
from .utils_optics import normalize
from .vector_paraxial_fields_XY import Vector_paraxial_field_XY


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
        self._type = 'Vector_paraxial_source_XY'

    def constant_wave(self,
                      u=1,
                      v=(1, 0),
                      has_normalization=False,
                      radius=0.):
        """Provides a constant polarization to a scalar_source_xy

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            v (float, float): polarization vector
            normalize (bool): If True, normalize polarization vector
        """

        self = define_initial_field(self, u)

        if has_normalization:
            v = normalize(v)

        self.Ex = v[0] * self.Ex
        self.Ey = v[1] * self.Ey

        self.mask_circle(radius=radius)

    def azimuthal_wave(self, u=1, r0=(0., 0.), radius=0.):
        """Provides a constant polarization to a scalar_source_xy

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0(float, float): center of rotation
            radius (float, float): Radius of a circular mask
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.sin(angle) * self.Ex
        self.Ey = -np.cos(angle) * self.Ey

        self.mask_circle(r0=r0, radius=radius)

    def radial_wave(self, u=1, r0=(0., 0.), radius=0.):
        """Provides a constant polarization to a scalar_source_xy

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0(float, float): center of rotation
            radius (float, float): Radius of a circular mask
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.cos(angle) * self.Ex
        self.Ey = np.sin(angle) * self.Ey

        self.mask_circle(r0=r0, radius=radius)

    def radial_inverse_wave(self, u=1, r0=(0., 0.), radius=0.):
        """Provides a constant polarization to a scalar_source_xy

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0(float, float): center of rotation
            radius (float, float): Radius of a circular mask
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.cos(angle) * self.Ex
        self.Ey = -np.sin(angle) * self.Ey

        self.mask_circle(r0=r0, radius=radius)

    def azimuthal_inverse_wave(self, u=1, r0=(0., 0.), radius=0.):
        """Provides a constant polarization to a scalar_source_xy

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0(float, float): center of rotation
            radius (float, float): Radius of a circular mask
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)

        self.Ex = np.sin(angle) * self.Ex
        self.Ey = np.cos(angle) * self.Ey

        self.mask_circle(r0=r0, radius=radius)

    def local_polarized_vector_wave(self,
                                    u=1,
                                    r0=(0., 0.),
                                    m=1,
                                    fi0=0,
                                    radius=0.):
        """"local radial polarized vector wave.


        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0 (float, float): r0 of beam
            m (integer): integer with order
            fi0 (float): initial phase
            radius (float, float): Radius of a circular mask

        References:
            Qwien Zhan 'Vectorial Optical Fields' page 33
        """

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        delta = m * angle + fi0

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

        self.mask_circle(r0=r0, radius=radius)

    def local_polarized_vector_wave_radial(self,
                                           u=1,
                                           r0=(0 * um, 0 * um),
                                           m=1,
                                           fi0=0,
                                           radius=0.):
        """local radial polarized vector wave.


        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0 (float, float): center of beam
            m (integer): integer with order
            fi0 (float): initial phase
            radius (float, float): Radius of a circular mask

        References:
            Qwien Zhan 'Vectorial Optial Fields' page 36
        """

        if radius == 0:
            radius_x = (self.x[-1] - self.x[0]) / 2
            radius_y = (self.y[-1] - self.y[0]) / 2
            radius = (radius_x, radius_y)

        elif isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        r = np.sqrt(vx**2 + vy**2)
        radius_0 = min(radius[0], radius[1])
        delta = 2 * m * np.pi * r / (radius_0 + eps) + fi0

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

        self.mask_circle(r0=r0, radius=radius)

    def local_polarized_vector_wave_hybrid(self,
                                           u=1,
                                           r0=(0 * um, 0 * um),
                                           m=1,
                                           n=1,
                                           fi0=0,
                                           radius=(0, 0)):
        """local hibrid polarized vector wave.
            Qwien Zhan 'Vectorial Optial Fields' page 36

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
            r0 (float, float): center of beam
            m (integer): integer with order
            n (integer): integer with order
            fi0 (float): initial phase
            radius (float, float): Radius of a circular mask
        """

        if radius == 0:
            radius_x = (self.x[-1] - self.x[0]) / 2
            radius_y = (self.y[-1] - self.y[0]) / 2
            radius = (radius_x, radius_y)

        elif isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        self = define_initial_field(self, u)

        vx = (self.X - r0[0])
        vy = (self.Y - r0[1])
        angle = np.arctan2(vy, vx)
        r = np.sqrt(vx**2 + vy**2)
        radius_0 = min(radius[0], radius[1])
        delta = m * angle + 2 * n * np.pi * r / (radius_0 + eps) + fi0

        self.Ex = self.Ex * np.cos(delta)
        self.Ey = self.Ey * np.sin(delta)

        self.mask_circle(r0=r0, radius=radius)

    def spiral_polarized_beam(self,
                              u=1,
                              r0=(0 * um, 0 * um),
                              alpha=0,
                              radius=(0, 0)):
        """Define spiral polarized beams:

        Parameters:
            u (Scalar_source_XY or np.complex): field to apply the polarization or constant value
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

        self.mask_circle(r0=r0, radius=radius)

    def mask_circle(self, r0=(0., 0.), radius=0.):
        """Mask vector field using a circular mask.

        Parameters:
            r0 (float, float): center of mask.
            radius (float, float): radius of mask
        """

        if radius == 0:
            radius_x = (self.x[-1] - self.x[0]) / 2
            radius_y = (self.y[-1] - self.y[0]) / 2
            radius = (radius_x, radius_y)

        elif radius in (None, '', []):
            return

        elif isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        if r0 in (0, None, '', []):
            r0_x = (self.x[-1] + self.x[0]) / 2
            r0_y = (self.y[-1] + self.y[0]) / 2
            r0 = (r0_x, r0_y)

        if radius[0] * radius[1] > 0:
            t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
            t1.circle(r0=r0, radius=radius, angle=0 * degrees)
            self.Ex = t1.u * self.Ex
            self.Ey = t1.u * self.Ey

    def to_py_pol(self):
        """Pass Ex, Ey field to py_pol package for software analysis
        """

        j0 = Jones_vector(name="from Diffractio")
        j0.from_components(Ex=self.Ex, Ey=self.Ey)

        return j0


def define_initial_field(EM, u=None):
    """Defines the initial field EM = (Ex, Ey) in terms of u.

        Parameters:
            EM (vector_paraxial_source_XY):
            u (scalar_source_XY, or None, or 1): if scalar_source it is written in Ex and Ey, is 1 Ex=1, Ey=1, if None, does nothing,
    """

    # check data size
    if isinstance(u, (float, int, complex)):
        EM.Ex = u * np.ones_like(EM.Ex)
        EM.Ey = u * np.ones_like(EM.Ey)
    elif isinstance(u, (Scalar_mask_XY, Scalar_field_XY, Scalar_source_XY)):
        EM.Ex = u.u
        EM.Ey = u.u
    if u in (0, None, '', []):
        EM.Ex = np.ones_like(EM.Ex)
        EM.Ey = np.ones_like(EM.Ey)

    return EM

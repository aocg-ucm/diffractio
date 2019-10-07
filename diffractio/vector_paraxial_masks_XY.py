# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_paraxial_mask_XY class for defining vector masks. Its parent is Vector_paraxial_field_XY.

The main atributes are:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field


*Class for bidimensional vector XY masks*

*Functions*
    * unique_masks
    * equal_masks
    * global_mask
    * complementary_masks
    * apply_polarization
    * polarizer_linear
    * quarter_waveplate
    * half_wave
    * polarizer_retarder
"""
from diffractio import degrees, np
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.vector_paraxial_fields_XY import Vector_paraxial_field_XY
from py_pol.jones_matrix import Jones_matrix


class Vector_paraxial_mask_XY(Vector_paraxial_field_XY):
    def __init__(self, x, y, wavelength, info=''):
        super(self.__class__, self).__init__(x, y, wavelength, info)

    def equal_masks(self, mask):
        """The same mask is applied to all the fields.

        Parameters:
            mask (scalar_mask_XY): mask to apply.

        """

        self.Ex = mask.u
        self.Ey = mask.u

    def unique_mask(self, mask, v=(1, 0)):
        """The same mask is applied to all the fields. Each field can have a different polarization, given in v.

        Parameters:
            masks (scalar_mask_XY): mask to apply.
            v (3x1 numpy.array): complex array with polarizations for each field.
                When v=(1,1,1) the field is not affected by polarization.

        Todo:
            Nothing is applied to H (everything is set to 0 Hx=Hy=Hz=0)
        """

        # cargo el field en las components
        self.Ex = v[0] * mask.u
        self.Ey = v[1] * mask.u

    def global_mask(self):
        """Applies a global circular/elliptical mask to all the fields. Data are obtained from self.x and self.y.
        """

        x_center = (self.x.max() + self.x.min()) / 2
        y_center = (self.y.max() + self.y.min()) / 2
        x_radius = (self.x.max() - self.x.min()) / 2
        y_radius = (self.y.max() - self.y.min()) / 2
        c = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        c.circle(
            r0=(x_center, y_center),
            radius=(x_radius, y_radius),
            angle=0 * degrees)
        self.Ex = self.Ex * c.u
        self.Ey = self.Ey * c.u

    def complementary_masks(self, mask, v1=(1, 0), v2=(0, 1)):
        """Creates two different fields Ex and Ey from a mask and its complementary.

        Parameters:
            mask (scalar_mask_XY): Mask preferently binary
            v1 (2x1 numpy.array): vector polarization of clear part of mask.
            v2 (2x1 numpy.array): vector polarization of dark part of mask.

        Warning:
            Mask should be binary. Else the function should binarize it: Todo.
        """

        v1 = v1 / np.sqrt(np.abs(v1[0])**2 + np.abs(v1[1])**2)
        v2 = v2 / np.sqrt(np.abs(v2[0])**2 + np.abs(v2[1])**2)

        t = np.abs(mask.u)**2
        t = t / t.max()

        self.Ex = v1[0] * t + v2[0] * (1 - t)
        self.Ey = v1[1] * t + v2[1] * (1 - t)

    def apply_polarization(self, polarizer):
        """Generates a constant polarization mask from py_pol polarization.Jones_matrix.
        This is the most general function to obtain polarizer.

        Parameters:
            polarizer (2x2 numpy.matrix): Jones_matrix

        Warning:
            Todo: Nothing is performed on H, only Ex, Ey.

        Todo:
            Check function. I do not understand well.
        """

        uno = np.ones_like(self.X, dtype=complex)
        zero = np.zeros_like(self.X, dtype=complex)
        polarizer = np.asarray(polarizer.M)

        Ex_new = uno * polarizer[0, 0] + uno * polarizer[0, 1]
        Ey_new = uno * polarizer[1, 0] + uno * polarizer[1, 1]
        self.Ex = Ex_new
        self.Ey = Ey_new

    def polarizer_linear(self, angle=0 * degrees):
        """Generates an XY linear polarizer.

        Parameters:
            angle (float): linear polarizer angle

        """
        PL = Jones_matrix('m0')
        PL.diattenuator_perfect(angle=angle)
        self.apply_polarization(PL)

    def quarter_waveplate(self, angle=0 * degrees):
        """Generates an XY quarter wave plate.

        Parameters:
            angle (float): polarizer angle

        """
        PL = Jones_matrix('m0')
        PL.quarter_waveplate(angle=angle)
        self.apply_polarization(PL)

    def half_waveplate(self, angle=0 * degrees):
        """Generates an XY half wave plate.

        Parameters:
            angle (float): polarizer angle

        """
        PL = Jones_matrix('m0')
        PL.half_waveplate(angle=angle)
        self.apply_polarization(PL)

    def polarizer_retarder(self,
                           delay=0 * degrees,
                           p1=1,
                           p2=1,
                           angle=0 * degrees):
        """Generates an XY retarder.

        Parameters:
            delay (float): delay between Ex and Ey components.
            p1 (float): transmittance of fast axis.
            p2 (float): transmittance of slow axis.
            angle (float): linear polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.diattenuator_retarder_linear(delay=delay, p1=p1, p2=p1, angle=angle)
        self.apply_polarization(PL)
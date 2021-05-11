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
    * from_py_pol
    * polarizer_linear
    * quarter_waveplate
    * half_wave
    * polarizer_retarder
"""
from py_pol.jones_matrix import Jones_matrix

from . import degrees, np, number_types, params_drawing, plt
from .scalar_masks_XY import Scalar_mask_XY
from .vector_paraxial_fields_XY import Vector_paraxial_field_XY
from .vector_paraxial_sources_XY import Vector_paraxial_source_XY


class Vector_paraxial_mask_XY(Vector_paraxial_field_XY):
    def __init__(self, x, y, wavelength, info=''):
        super(self.__class__, self).__init__(x, y, wavelength, info)
        self._type = 'Vector_paraxial_mask_XY'

        self.M00 = np.zeros_like(self.X)
        self.M01 = np.zeros_like(self.X)
        self.M10 = np.zeros_like(self.X)
        self.M11 = np.zeros_like(self.X)

        del self.Ex, self.Ey

    def __mul__(self, other):
        """
        Multilies the Vector_paraxial_mask_XY matrix by another Vector_paraxial_mask_XY.

        Parameters:
            other (Vector_paraxial_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_paraxial_mask_XY): Result.
        """

        if isinstance(other, number_types):
            m3 = Vector_paraxial_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other

        elif other._type in ('Vector_paraxial_mask_XY', 'Vector_paraxial_field_XY'):
            m3 = Vector_paraxial_mask_XY(self.x, self.y, self.wavelength)

            m3.M00 = other.M00 * self.M00 + other.M01 * self.M10
            m3.M01 = other.M00 * self.M01 + other.M01 * self.M11
            m3.M10 = other.M10 * self.M00 + other.M11 * self.M10
            m3.M11 = other.M10 * self.M01 + other.M11 * self.M11

        else:
            raise ValueError('other thype ({}) is not correct'.format(
                type(other)))

        return m3

    def __rmul__(self, other):
        """
        Multilies the Vector_paraxial_mask_XY matrix by another Vector_paraxial_mask_XY.

        Parameters:
            other (Vector_paraxial_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_paraxial_mask_XY): Result.
        """
        if isinstance(other, number_types):
            m3 = Vector_paraxial_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other
            # print("numero * matriz")

        elif other._type in ('Vector_paraxial_source_XY',
                             'Vector_paraxial_field_XY'):
            m3 = Vector_paraxial_source_XY(self.x, self.y, self.wavelength)
            m3.Ex = self.M00 * other.Ex + self.M01 * other.Ey
            m3.Ey = self.M10 * other.Ex + self.M11 * other.Ey

        return m3

    def rotate(self, angle, new_mask=False):
        """Rotates the mask a certain angle.abs

        Parameters:
            angle (float): rotation angle in radians
            new_mask (bool): if True generates a new mask

        Returns:
            if new_mask is True: Vector_paraxial_mask_XY
        """

        # TODO:
        # como no quiero hacerlo como en pypol hay que sacar la funcion analitica

        pass

    def apply_circle(self, r0=None, radius=None):
        """The same circular mask is applied to all the Jones Matrix.

        Parameters:
            r0 (float, float): center, if None it is generated
            radius (float, float): radius, if None it is generated
        """
        if radius is None:
            x_min, x_max = self.x[0], self.x[-1]
            y_min, y_max = self.y[0], self.y[-1]

            x_radius, y_radius = (x_max - x_min) / 2, (y_max - y_min) / 2

            radius = (x_radius, y_radius)

        if r0 is None:
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            r0 = (x_center, y_center)

        u_mask_circle = Scalar_mask_XY(self.x, self.y, self.wavelength)
        u_mask_circle.circle(r0=r0, radius=radius)

        self.M00 = self.M00 * u_mask_circle.u
        self.M01 = self.M01 * u_mask_circle.u
        self.M10 = self.M10 * u_mask_circle.u
        self.M11 = self.M11 * u_mask_circle.u

    def apply_scalar_mask(self, u_mask):
        """The same mask u_mask is applied to all the Jones Matrix.

        Parameters:
            u_mask (scalar_mask_XY): mask to apply.

        """
        self.M00 = self.M00 * u_mask.u
        self.M01 = self.M01 * u_mask.u
        self.M10 = self.M10 * u_mask.u
        self.M11 = self.M11 * u_mask.u

    def complementary_masks(self,
                            mask,
                            state_0=np.array([[1, 0], [0, 0]]),
                            state_1=np.array([[0, 0], [0, 1]]),
                            is_binarized=True):
        """Creates two different fields Ex and Ey from a mask and its complementary.
        For generality, is mask is a decimal number between 0 and 1, it takes the linear interpolation.

        Parameters:
            mask (scalar_mask_XY): Mask preferently binary. if not, it is binarized
            state_0 (2x1 numpy.array): polarization matrix for 0s.
            state_1 (2x1 numpy.array): polarization matrix for 1s.

        Warning:
            TODO: Mask should be binary. Else the function should binarize it.
        """

        t = np.abs(mask.u)**2
        if is_binarized:
            t = t / t.max()
            t[t < 0.5] = 0
            t[t >= 0.5] = 1

        self.M00 = t * state_1[0, 0] + (1 - t) * state_0[0, 0]
        self.M01 = t * state_1[0, 1] + (1 - t) * state_0[1, 0]
        self.M10 = t * state_1[1, 0] + (1 - t) * state_0[0, 1]
        self.M11 = t * state_1[1, 1] + (1 - t) * state_0[1, 1]

    def from_py_pol(self, polarizer):
        """Generates a constant polarization mask from py_pol polarization.Jones_matrix.
        This is the most general function to obtain a polarizer.

        Parameters:
            polarizer (2x2 numpy.matrix): Jones_matrix
        """

        if isinstance(polarizer, Jones_matrix):
            M = polarizer.M
        else:
            M = polarizer

        uno = np.ones_like(self.X, dtype=complex)
        M = np.asarray(M)

        self.M00 = uno * M[0, 0]
        self.M01 = uno * M[0, 1]
        self.M10 = uno * M[1, 0]
        self.M11 = uno * M[1, 1]

    def polarizer_linear(self, azimuth=0 * degrees):
        """Generates an XY linear polarizer.

        Parameters:
            angle (float): linear polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.diattenuator_perfect(azimuth=azimuth)
        self.from_py_pol(PL)

    def quarter_waveplate(self, azimuth=0 * degrees):
        """Generates an XY quarter wave plate.

        Parameters:
            azimuth (float): polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.quarter_waveplate(azimuth=azimuth)
        self.from_py_pol(PL)

    def half_waveplate(self, azimuth=0 * degrees):
        """Generates an XY half wave plate.

        Parameters:
            azimuth (float): polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.half_waveplate(azimuth=azimuth)
        self.from_py_pol(PL)

    def polarizer_retarder(self,
                           R=0 * degrees,
                           p1=1,
                           p2=1,
                           azimuth=0 * degrees):
        """Generates an XY retarder.

        Parameters:
            R (float): retardance between Ex and Ey components.
            p1 (float): transmittance of fast axis.
            p2 (float): transmittance of slow axis.
            azimuth (float): linear polarizer angle
        """
        PL = Jones_matrix('m0')
        PL.diattenuator_retarder_linear(
            R=R, p1=p1, p2=p1, azimuth=azimuth)
        self.from_py_pol(PL)

    def to_py_pol(self):
        """Pass mask to py_pol.jones_matrix

        Returns:
            py_pol.jones_matrix

        """

        m0 = Jones_matrix(name="from Diffractio")
        m0.from_components((self.M00, self.M01, self.M10, self.M11))

        return m0

    def draw(self, kind='amplitude', z_scale='um'):
        """Draws the mask. It must be different to sources.

        Parameters:
            kind (str): 'amplitude', 'amplitudes', 'phase', 'phases', 'all'
        """
        # def draw_masks(self, kind='fields'):

        extension = np.array([self.x[0], self.x[-1], self.y[0], self.y[-1]])
        if z_scale == 'mm':
            extension = extension / 1000.

        if kind in ('amplitude', 'all'):
            plt.figure()
            plt.set_cmap(params_drawing['color_intensity'])
            fig, axs = plt.subplots(
                2,
                2,
                sharex='col',
                sharey='row',
                gridspec_kw={
                    'hspace': 0.05,
                    'wspace': 0.05
                })
            im1 = axs.flat[0].imshow(np.abs(self.M00), extent=extension)
            im1.set_clim(0, 1)
            im1 = axs.flat[1].imshow(np.abs(self.M01), extent=extension)
            im1.set_clim(0, 1)
            im1 = axs.flat[2].imshow(np.abs(self.M10), extent=extension)
            im1.set_clim(0, 1)
            im1 = axs.flat[3].imshow(np.abs(self.M11), extent=extension)
            im1.set_clim(0, 1)

            plt.suptitle("Amplitudes")
            cax = plt.axes([.95, 0.15, 0.05, 0.7])
            plt.colorbar(im1, cax=cax)
            if z_scale == 'um':
                axs.flat[2].set_xlabel(r'x ($\mu$m)')
                axs.flat[2].set_ylabel(r'y($\mu$m)')
            elif z_scale == 'mm':
                axs.flat[2].set_xlabel(r'x (mm)')
                axs.flat[2].set_ylabel(r'y (mm)')

        if kind in ('phase', 'all'):

            plt.figure()
            plt.set_cmap(params_drawing['color_phase'])

            fig, axs = plt.subplots(
                2,
                2,
                sharex='col',
                sharey='row',
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0.1
                })
            im1 = axs.flat[0].imshow(
                np.angle(self.M00) / degrees, extent=extension)
            im1.set_clim(-180, 180)
            im1 = axs.flat[1].imshow(
                np.angle(self.M01) / degrees, extent=extension)
            im1.set_clim(-180, 180)
            im1 = axs.flat[2].imshow(
                np.angle(self.M10) / degrees, extent=extension)
            im1.set_clim(-180, 180)
            im1 = axs.flat[3].imshow(
                np.angle(self.M11) / degrees, extent=extension)
            im1.set_clim(-180, 180)
            plt.suptitle("phases")
            cax = plt.axes([.95, 0.15, 0.05, 0.7])
            plt.colorbar(im1, cax=cax)

            plt.set_cmap(params_drawing['color_intensity'])


def rotation_matrix_Jones(angle):
    """Creates an array of Jones 2x2 rotation matrices.

    Parameters:
        angle (np.array): array of angle of rotation, in radians.

    Returns:
        numpy.array: 2x2 matrix
    """
    M = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    return M

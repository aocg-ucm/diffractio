# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_mask_XY class for defining vector masks. Its parent is Vector_field_XY.

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
import copy

from py_pol.jones_matrix import Jones_matrix

from . import degrees, np, number_types, plt
from .config import CONF_DRAWING
from .scalar_masks_XY import Scalar_mask_XY
from .utils_optics import field_parameters
from .vector_fields_XY import Vector_field_XY
from .vector_sources_XY import Vector_source_XY


class Vector_mask_XY(Vector_field_XY):

    def __init__(self, x, y, wavelength, info=''):
        super(self.__class__, self).__init__(x, y, wavelength, info)
        self._type = 'Vector_mask_XY'

        self.M00 = np.zeros_like(self.X, dtype=complex)
        self.M01 = np.zeros_like(self.X, dtype=complex)
        self.M10 = np.zeros_like(self.X, dtype=complex)
        self.M11 = np.zeros_like(self.X, dtype=complex)

        del self.Ex, self.Ey, self.Ez

    def __add__(self, other, kind='standard'):
        """adds two Vector_mask_XY. For example two  masks

        Parameters:
            other (Vector_mask_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_mask_XY: `M3 = M1 + M2`
        """

        if other._type in ('Vector_mask_XY'):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)

            m3.M00 = other.M00 + self.M00
            m3.M01 = other.M01 + self.M01
            m3.M10 = other.M10 + self.M10
            m3.M11 = other.M11 + self.M11

        return m3

    def __mul__(self, other):
        """
        Multilies the Vector_mask_XY matrix by another Vector_mask_XY.

        Parameters:
            other (Vector_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_mask_XY): Result.
        """

        if isinstance(other, number_types):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other

        elif other._type in ('Vector_mask_XY', 'Vector_field_XY'):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)

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
        Multilies the Vector_mask_XY matrix by another Vector_mask_XY.

        Parameters:
            other (Vector_mask_XY): 2nd object to multiply.

        Returns:
            v_mask_XY (Vector_mask_XY): Result.
        """
        if isinstance(other, number_types):
            m3 = Vector_mask_XY(self.x, self.y, self.wavelength)
            m3.M00 = self.M00 * other
            m3.M01 = self.M01 * other
            m3.M10 = self.M10 * other
            m3.M11 = self.M11 * other
            # print("numero * matriz")

        elif other._type in ('Vector_source_XY', 'Vector_field_XY'):
            m3 = Vector_source_XY(self.x, self.y, self.wavelength)
            m3.Ex = self.M00 * other.Ex + self.M01 * other.Ey
            m3.Ey = self.M10 * other.Ex + self.M11 * other.Ey

        return m3

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    # def rotate(self, angle, new_mask=False):
    #     """Rotates the mask a certain angle.abs
    #
    #     Parameters:
    #         angle (float): rotation angle in radians
    #         new_mask (bool): if True generates a new mask
    #
    #     Returns:
    #         if new_mask is True: Vector_mask_XY
    #     """
    #
    #     # TODO:
    #     # como no quiero hacerlo como en pypol hay que sacar la funcion analitica
    #
    #     pass

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

    def pupil(self, r0=None, radius=None, angle=0 * degrees):
        """place a pupil in the mask. If r0 or radius are None, they are computed using the x,y parameters.

        Parameters:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            pupil(r0=(0 * um, 0 * um), radius=(250*um, 125*um), angle=0*degrees)
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
        pupil0 = np.zeros(np.shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + \
            (Yrot)**2 / (radiusy**2 + 1e-15) < 1
        pupil0[ipasa] = 1
        self.M00 = self.M00 * pupil0
        self.M01 = self.M01 * pupil0
        self.M10 = self.M10 * pupil0
        self.M11 = self.M11 * pupil0

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
        """Creates a vector mask from a scalar mask. It assign an state_0 to 0 values and a state_1 to 1 values..
        For generality, ik mask is a decimal number between 0 and 1, it takes the linear interpolation.

        Parameters:
            mask (scalar_mask_XY): Mask preferently binary. if not, it is binarized
            state_0 (2x2 numpy.array): Jones matrix for 0s.
            state_1 (2x2 numpy.array): Jones matrix for 1s.

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

    def multilevel_mask(self, mask, states, discretize=True, normalize=True):
        """Generates a multilevel vector mask, based in a scalar_mask_XY. The levels should be integers in amplitude (0,1,..., N).
            If it is not like this, discretize generates N levels.
            Usually masks are 0-1. Then normalize generates levels 0-N.

            Parameters:
                mask (scalar_mask_XY): 0-N discrete scalar mask.
                states (np.array or Jones_matrix): Jones matrices to assign to each level
                discretize (bool): If True, a continuous mask is converted to N levels.
                normalize (bool): If True, levels are 0,1,.., N.

        """
        mask_new = mask.duplicate()

        num_levels = len(states)

        if discretize is True:
            mask_new.discretize(num_levels=num_levels, new_field=False)

        if normalize is True:
            mask_new.u = mask_new.u / mask_new.u.max()
            mask_new.u = mask_new.u * num_levels - 0.5

        mask_new.u = np.real(mask_new.u)
        mask_new.u = mask_new.u.astype(np.int)

        for i, state in enumerate(states):
            # print(state)
            i_level = (mask_new.u == i)
            self.M00[i_level] = state.M[0, 0, 0]
            self.M01[i_level] = state.M[0, 1, 0]
            self.M11[i_level] = state.M[1, 1, 0]
            self.M10[i_level] = state.M[1, 0, 0]
            # print(self.M00[i_level][0], self.M01[i_level][0], self.M10[i_level][0], self.M11[i_level][0])

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
        PL.diattenuator_retarder_linear(R=R, p1=p1, p2=p1, azimuth=azimuth)
        self.from_py_pol(PL)

    def to_py_pol(self):
        """Pass mask to py_pol.jones_matrix

        Returns:
            py_pol.jones_matrix

        """

        m0 = Jones_matrix(name="from Diffractio")
        m0.from_components((self.M00, self.M01, self.M10, self.M11))
        m0.shape = self.M00.shape

        return m0

    def draw(self, kind='amplitude', range_scale='um'):
        """Draws the mask. It must be different to sources.

        Parameters:
            kind (str): 'amplitude', 'phase', 'all'
        """
        # def draw_masks(self, kind='fields'):

        extension = np.array([self.x[0], self.x[-1], self.y[0], self.y[-1]])
        if range_scale == 'mm':
            extension = extension / 1000.

        a00, int00, phase00 = field_parameters(self.M00,
                                               has_amplitude_sign=False)

        a01, int01, phase01 = field_parameters(self.M01,
                                               has_amplitude_sign=False)
        a10, int10, phase10 = field_parameters(self.M10,
                                               has_amplitude_sign=False)
        a11, int11, phase11 = field_parameters(self.M11,
                                               has_amplitude_sign=False)

        a_max = np.abs((a00, a01, a10, a11)).max()

        if kind in ('amplitude', 'all'):
            plt.set_cmap(CONF_DRAWING['color_intensity'])
            fig, axs = plt.subplots(2,
                                    2,
                                    sharex='col',
                                    sharey='row',
                                    gridspec_kw={
                                        'hspace': 0.25,
                                        'wspace': 0.025
                                    })
            im1 = axs[0, 0].imshow(a00, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(a01, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(a10, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(a11, extent=extension, origin='lower')
            im1.set_clim(0, a_max)
            axs[1, 1].set_title("J11")

            plt.suptitle("Amplitudes", fontsize=20)
            cax = plt.axes([.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

            if range_scale == 'um':
                axs[1, 0].set_xlabel(r'x ($\mu$m)')
                axs[1, 0].set_ylabel(r'y($\mu$m)')
            elif range_scale == 'mm':
                axs[1, 0].set_xlabel(r'x (mm)')
                axs[1, 0].set_ylabel(r'y (mm)')

        if kind in ('phase', 'all'):
            plt.set_cmap(CONF_DRAWING['color_phase'])

            fig, axs = plt.subplots(2,
                                    2,
                                    sharex='col',
                                    sharey='row',
                                    gridspec_kw={
                                        'hspace': 0.25,
                                        'wspace': 0.00
                                    })
            im1 = axs[0, 0].imshow(np.angle(self.M00) / degrees,
                                   extent=extension,
                                   origin='lower')
            im1.set_clim(-180, 180)
            axs[0, 0].set_title("J00")

            im1 = axs[0, 1].imshow(np.angle(self.M01) / degrees,
                                   extent=extension,
                                   origin='lower')
            im1.set_clim(-180, 180)
            axs[0, 1].set_title("J01")

            im1 = axs[1, 0].imshow(np.angle(self.M10) / degrees,
                                   extent=extension,
                                   origin='lower')
            im1.set_clim(-180, 180)
            axs[1, 0].set_title("J10")

            im1 = axs[1, 1].imshow(np.angle(self.M11) / degrees,
                                   extent=extension,
                                   origin='lower')
            im1.set_clim(-180, 180)
            axs[1, 1].set_title("J11")

            plt.suptitle("Phases", fontsize=20)
            cax = plt.axes([.89, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im1, cax=cax)
            cbar.set_ticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])

            if range_scale == 'um':
                axs[1, 0].set_xlabel(r'x ($\mu$m)')
                axs[1, 0].set_ylabel(r'y($\mu$m)')
            elif range_scale == 'mm':
                axs[1, 0].set_xlabel(r'x (mm)')
                axs[1, 0].set_ylabel(r'y (mm)')


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

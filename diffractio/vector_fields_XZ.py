# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_field_X class. It is required also for generating masks and fields.
The main atributes are:
    * self.x - x positions of the field
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.Ez - z component of electric field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for X vector fields*

*Definition of a scalar field*
    * add, substract fields
    * save, load data, clean, get, normalize
    * cut_resample

*Vector parameters*
    * polarization_states

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""

import copy
from matplotlib import rcParams
from scipy.interpolate import RectBivariateSpline

from . import degrees, eps, mm, np, plt
from .config import CONF_DRAWING
from .scalar_fields_XZ import Scalar_field_XZ
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw, reduce_matrix_size
from .utils_math import nearest
from .utils_optics import normalize_field

percentage_intensity = CONF_DRAWING['percentage_intensity']


class Vector_field_XZ(object):
    """Class for vectorial fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
        self.Ez (numpy.array): Electric_z field
    """

    def __init__(self, x, z, wavelength, info=''):
        self.x = x
        self.z = z
        self.wavelength = wavelength  # la longitud de onda

        self.X, self.Z = np.meshgrid(x, z)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Vector_field_XZ'
        self.info = info
        self.date = get_date()

    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print("{}\n - x:  {},     Ex:  {}".format(self.type, self.x.shape,
                                                  self.Ex.shape))
        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um"
            .format(self.x[0], self.x[-1], self.x[1] - self.x[0]))
        print(
            " - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um"
            .format(self.z[0], self.z[-1], self.x[1] - self.z[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""

    def __add__(self, other, kind='standard'):
        """adds two Vector_field_X. For example two light sources or two masks

        Parameters:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_X: `E3 = E1 + E2`
        """

        EM = Vector_field_XZ(self.x, self.z, self.wavelength)

        if kind == 'standard':
            EM.Ex = self.Ex + other.Ex
            EM.Ey = self.Ey + other.Ey
            EM.Ez = self.Ez + other.Ez

        return EM

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
        """Load data from a file to a Vector_field_X.
            The methods included are: npz, matlab

        Parameters:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

        if verbose is True:
            print(dict0.keys())

    def clear_field(self):
        """Removes the fields Ex, Ey, Ez"""

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ey, dtype=complex)
        self.Ez = np.zeros_like(self.Ez, dtype=complex)

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def get(self, kind='fields', is_matrix=True):
        """Takes the vector field and divide in Scalar_field_X.

        Parameters:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Vector_field_X: (Ex, Ey, Ez),
        """

        self.Ex = self.Ex
        self.Ey = self.Ey
        self.Ez = self.Ez

        if kind == 'fields':
            if is_matrix:
                return self.Ex, self.Ey, self.Ez

            else:
                Ex = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ex.u = self.Ex
                Ey = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ey.u = self.Ey
                Ez = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ez.u = self.Ez
                return Ex, Ey, Ez

        elif kind == 'intensity':
            intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(
                self.Ez)**2

            if is_matrix:
                return intensity

            else:
                Intensity = Scalar_field_XZ(x=self.x,
                                            z=self.z,
                                            wavelength=self.wavelength)
                Intensity.u = np.sqrt(intensity)

                return Intensity

        elif kind == 'intensities':
            intensity_x = np.abs(self.Ex)**2
            intensity_y = np.abs(self.Ey)**2
            intensity_z = np.abs(self.Ez)**2
            return intensity_x, intensity_y, intensity_z

        elif kind == 'phases':
            phase_x = np.angle(self.Ex)
            phase_y = np.angle(self.Ey)
            phase_z = np.angle(self.Ez)

            if is_matrix:
                return phase_x, phase_y, phase_z
            else:
                Ex = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ex.u = np.exp(1j * phase_x)
                Ey = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ey.u = np.exp(1j * phase_y)
                Ez = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
                Ez.u = np.exp(1j * phase_z)
                return Ex, Ey, Ez

        elif kind == 'stokes':
            # S0, S1, S2, S3
            return self.polarization_states(matrix=True)

        elif kind == 'params_ellipse':
            # A, B, theta, h
            return self.polarization_ellipse(pol_state=None, matrix=True)

        else:
            print("The parameter '{}'' in .get(kind='') is wrong".format(kind))

    def apply_mask(self, u):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Parameters:
           u (Scalar_mask_X): mask
         """
        self.Ex = self.Ex * u.u
        self.Ey = self.Ey * u.u
        self.Ez = self.Ez * u.u

    def intensity(self):
        """"Returns intensity.
        """
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(
            self.Ez)**2

        return intensity

    def polarization_states(self, matrix=False):
        """returns the Stokes parameters

        Parameters:
            Matrix (bool): if True returns Matrix, else Scalar_field_X

        Returns:
            S0,S1,S2,S3 images for Matrix=True
            S0,S1,S2,S3  for Matrix=False
        """

        I = np.abs(self.Ex)**2 + np.abs(self.Ey)**2
        Q = np.abs(self.Ex)**2 - np.abs(self.Ey)**2
        U = 2 * np.real(self.Ex * np.conjugate(self.Ey))
        V = 2 * np.imag(self.Ex * np.conjugate(self.Ey))

        if matrix is True:
            return I, Q, U, V
        else:

            CI = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
            CQ = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
            CU = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
            CV = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)

            CI.u = I
            CQ.u = Q
            CU.u = U
            CV.u = V

            return CI, CQ, CU, CV

    def polarization_ellipse(self, pol_state=None, matrix=False):
        """returns A, B, theta, h polarization parameter of elipses

        Parameters:
            pol_state (None or (I, Q, U, V) ): Polarization state previously computed
            Matrix (bool): if True returns Matrix, else Scalar_field_X

        Returns:
            A, B, theta, h for Matrix=True
            CA, CB, Ctheta, Ch for Matrix=False
        """
        if pol_state is None:
            I, Q, U, V = self.polarization_states(matrix=True)
        else:
            I, Q, U, V = pol_state
            I = I.u
            Q = Q.u
            U = U.u
            V = V.u

        Ip = np.sqrt(Q**2 + U**2 + V**2)
        L = Q + 1.j * U + eps

        A = np.real(np.sqrt(0.5 * (Ip + np.abs(L) + eps)))
        B = np.real(np.sqrt(0.5 * (Ip - np.abs(L) + eps)))
        theta = 0.5 * np.angle(L)
        h = np.sign(V + eps)

        if matrix is True:
            return A, B, theta, h
        else:
            CA = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
            CB = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
            Ctheta = Scalar_field_XZ(x=self.x,
                                     z=self.z,
                                     wavelength=self.wavelength)
            Ch = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)

            CA.u = A
            CB.u = B
            Ctheta.u = theta
            Ch.u = h
            return (CA, CB, Ctheta, Ch)

    def normalize(self, new_field=False):
        """Normalizes the field so that intensity.max()=1.

        Parameters:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, new_field)

    def draw(self,
             kind='intensity',
             logarithm=0,
             normalize=False,
             cut_value=None,
             filename='',
             draw=True,
             **kwargs):
        """Draws electromagnetic field

        Parameters:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            filename (str): if not '' stores drawing in file,

        """

        if draw is True:
            if kind == 'intensity':
                id_fig = self.__draw_intensity__(logarithm, normalize,
                                                 cut_value, **kwargs)
            elif kind == 'intensities':
                id_fig = self.__draw_intensities__(logarithm, normalize,
                                                   cut_value, **kwargs)

            elif kind == 'phases':
                id_fig = self.__draw_phases__(logarithm, normalize, cut_value,
                                              **kwargs)

            elif kind == 'fields':
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value,
                                              **kwargs)

            elif kind == 'stokes':
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value,
                                              **kwargs)

            elif kind == 'param_ellipses':
                id_fig = self.__draw_param_ellipse__(logarithm, normalize,
                                                     cut_value, **kwargs)

            else:
                print("not good kind parameter in vector_fields_X.draw()")
                id_fig = None

            if not filename == '':
                plt.savefig(filename,
                            dpi=300,
                            bbox_inches='tight',
                            pad_inches=0.1)

            return id_fig

    def __draw_intensity__(self,
                           logarithm,
                           normalize,
                           cut_value,
                           only_image=False,
                           color_intensity=CONF_DRAWING['color_intensity']):
        """Draws the intensity

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get('intensity')

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.z,
                                       intensity)

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.figure()
        h1 = plt.subplot(1, 1, 1)
        self.__draw1__(intensity, color_intensity, "", only_image=only_image)
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return h1

    def __draw_intensities__(self,
                             logarithm,
                             normalize,
                             cut_value,
                             only_image=False,
                             color_intensity=CONF_DRAWING['color_intensity']):
        """internal funcion: draws phase

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        tx, ty = rcParams['figure.figsize']

        intensity1 = np.abs(self.Ex)**2
        intensity1 = normalize_draw(intensity1, logarithm, normalize,
                                    cut_value)

        intensity2 = np.abs(self.Ey)**2
        intensity2 = normalize_draw(intensity2, logarithm, normalize,
                                    cut_value)

        intensity3 = np.abs(self.Ez)**2
        intensity3 = normalize_draw(intensity3, logarithm, normalize,
                                    cut_value)

        intensity_max = np.max(
            (intensity1.max(), intensity2.max(), intensity3.max()))

        percentage_z = 0.01

        if intensity3.max() < percentage_z * intensity_max:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            self.__draw1__(intensity1,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 2, 2)
            self.__draw1__(intensity2,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2
        else:

            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            self.__draw1__(intensity1,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 3, 2)
            self.__draw1__(intensity2,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(0, intensity_max)

            h3 = plt.subplot(1, 3, 3)
            self.__draw1__(intensity3,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_phases__(self,
                        logarithm,
                        normalize,
                        cut_value,
                        only_image=False,
                        color_intensity=CONF_DRAWING['color_intensity']):
        """internal funcion: draws phase

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        tx, ty = rcParams['figure.figsize']

        intensity1 = np.abs(self.Ex)**2
        intensity1 = normalize_draw(intensity1, logarithm, normalize,
                                    cut_value)

        intensity2 = np.abs(self.Ey)**2
        intensity2 = normalize_draw(intensity2, logarithm, normalize,
                                    cut_value)

        intensity3 = np.abs(self.Ez)**2
        intensity3 = normalize_draw(intensity3, logarithm, normalize,
                                    cut_value)

        intensity_max = np.max(
            (intensity1.max(), intensity2.max(), intensity3.max()))

        percentage_z = 0.01

        if intensity3.max() < percentage_z * intensity_max:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex)**2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 2, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey)**2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(-180, 180)
            plt.tight_layout()

            return h1, h2
        else:

            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex)**2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 3, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey)**2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(-180, 180)

            h3 = plt.subplot(1, 3, 3)
            phase = np.angle(self.Ez)
            intensity = np.abs(self.Ez)**2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees,
                           color_intensity,
                           "",
                           only_image=only_image)
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0,
                                bottom=0,
                                right=1,
                                top=1,
                                wspace=0.05,
                                hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_fields__(self,
                        logarithm,
                        normalize,
                        cut_value,
                        color_intensity=CONF_DRAWING['color_intensity'],
                        color_phase=CONF_DRAWING['color_phase']):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        intensity_x = np.abs(self.Ex)**2
        intensity_x = normalize_draw(intensity_x, logarithm, normalize,
                                     cut_value)

        intensity_y = np.abs(self.Ey)**2
        intensity_y = normalize_draw(intensity_y, logarithm, normalize,
                                     cut_value)

        intensity_max = np.max((intensity_x.max(), intensity_y.max()))
        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        h1 = plt.subplot(2, 2, 1)

        __draw1__(self, intensity_x, "$I_x$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        __draw1__(self, intensity_y, "$I_y$")
        plt.clim(0, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        phase = np.angle(self.Ex)
        phase[intensity_x < percentage_intensity * (intensity_x.max())] = 0

        __draw1__(self, phase / degrees, color_phase, "$\phi_x$")
        plt.clim(-180, 180)

        h4 = plt.subplot(2, 2, 4)
        phase = np.angle(self.Ey)
        phase[intensity_y < percentage_intensity * (intensity_y.max())] = 0

        __draw1__(self, phase / degrees, color_phase, "$\phi_y$")
        plt.clim(-180, 180)
        h4 = plt.gca()
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return h1, h2, h3, h4

    def __draw_stokes__(self,
                        logarithm,
                        normalize,
                        cut_value,
                        color_intensity=CONF_DRAWING['color_intensity'],
                        color_stokes=CONF_DRAWING['color_stokes']):
        """__internal__: computes and draws CI, CQ, CU, CV parameters
        """

        tx, ty = rcParams['figure.figsize']

        S0, S1, S2, S3 = self.polarization_states(matrix=True)
        S0 = normalize_draw(S0, logarithm, normalize, cut_value)
        S1 = normalize_draw(S1, logarithm, normalize, cut_value)
        S2 = normalize_draw(S2, logarithm, normalize, cut_value)
        S3 = normalize_draw(S3, logarithm, normalize, cut_value)

        intensity_max = S0.max()

        plt.figure(figsize=(2 * tx, 2 * ty))
        h1 = plt.subplot(2, 2, 1)
        self.__draw1__(S0, color_intensity, "$S_0$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(S1, color_stokes, "$S_1$")
        plt.clim(-intensity_max, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        self.__draw1__(S2, color_stokes, "$S_2$")
        plt.clim(-intensity_max, intensity_max)

        h4 = plt.subplot(2, 2, 4)
        self.__draw1__(S3, color_stokes, "$S_3$")
        plt.clim(-intensity_max, intensity_max)

        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0.05,
                            hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_param_ellipse__(self,
                               color_intensity=CONF_DRAWING['color_intensity'],
                               color_phase=CONF_DRAWING['color_phase']):
        """__internal__: computes and draws polariations ellipses
        """
        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.z, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.z, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.z, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.z, h)

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        max_intensity = max(A.max(), B.max())

        h1 = plt.subplot(2, 2, 1)
        self.__draw1__(A, color_intensity, "$A$")
        plt.clim(0, max_intensity)
        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(B, color_intensity, "$B$")
        plt.clim(0, max_intensity)

        h3 = plt.subplot(2, 2, 3)
        self.__draw1__(theta / degrees, color_phase, "$\phi$")
        plt.clim(-180, 180)
        h4 = plt.subplot(2, 2, 4)
        self.__draw1__(h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_ellipses__(self,
                          logarithm=False,
                          normalize=False,
                          cut_value='',
                          num_ellipses=(21, 21),
                          amplification=0.75,
                          color_line='w',
                          line_width=1,
                          draw_arrow=True,
                          head_width=2,
                          ax=False,
                          color_intensity=CONF_DRAWING['color_intensity']):
        """__internal__: draw ellipses

        Parameters:
            num_ellipses (int): number of ellipses for parameters_ellipse
        """

        percentage_intensity = CONF_DRAWING['percentage_intensity']
        intensity_max = (np.abs(self.Ex)**2 + np.abs(self.Ey)**2).max()

        Dx = self.x[-1] - self.x[0]
        Dy = self.z[-1] - self.z[0]
        size_x = Dx / (num_ellipses[0])
        size_y = Dy / (num_ellipses[1])
        x_centers = size_x / 2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y / 2 + size_y * np.array(range(0, num_ellipses[1]))

        num_x, num_y = len(self.x), len(self.z)
        ix_centers = num_x / (num_ellipses[0])
        iy_centers = num_y / (num_ellipses[1])

        ix_centers = (np.round(
            ix_centers / 2 +
            ix_centers * np.array(range(0, num_ellipses[0])))).astype('int')
        iy_centers = (np.round(
            iy_centers / 2 +
            iy_centers * np.array(range(0, num_ellipses[1])))).astype('int')

        Ix_centers, Iy_centers = np.meshgrid(ix_centers.astype('int'),
                                             iy_centers.astype('int'))

        verbose = False
        if verbose is True:
            print(num_x, num_y, ix_centers, iy_centers)
            print(Dx, Dy, size_x, size_y)
            print(x_centers, y_centers)
            print(Ix_centers, Iy_centers)

        E0x = self.Ex[Iy_centers, Ix_centers]
        E0y = self.Ey[Iy_centers, Ix_centers]

        angles = np.linspace(0, 360 * degrees, 64)

        if ax is False:
            self.draw('intensity',
                      logarithm=logarithm,
                      color_intensity=color_intensity)
            ax = plt.gca()

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iy_centers):
                Ex = np.real(E0x[j, i] * np.exp(1j * angles))
                Ey = np.real(E0y[j, i] * np.exp(1j * angles))

                max_r = np.sqrt(np.abs(Ex)**2 + np.abs(Ey)**2).max()
                size_dim = min(size_x, size_y)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:

                    Ex = Ex / max_r * size_dim * amplification / 2 + (
                        +self.x[int(xi)])
                    Ey = Ey / max_r * size_dim * amplification / 2 + self.z[
                        int(yj)]

                    ax.plot(Ex, Ey, color_line, lw=line_width)
                    if draw_arrow:
                        ax.arrow(Ex[0],
                                 Ey[0],
                                 Ex[0] - Ex[1],
                                 Ey[0] - Ey[1],
                                 width=0,
                                 head_width=head_width,
                                 fc=color_line,
                                 ec=color_line,
                                 length_includes_head=False)
                # else:
                #     print(max_r, intensity_max,
                #           percentage_intensity * intensity_max)

    def __draw1__(self,
                  image,
                  colormap,
                  title='',
                  has_max=False,
                  only_image=False):
        """Draws image

        Parameters:
            image (numpy.array): array with drawing
            colormap (str): colormap
            title (str): title of drawing
        """
        extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

        h = plt.imshow(image,
                       interpolation='bilinear',
                       aspect='auto',
                       origin='lower',
                       extent=extension)
        h.set_cmap(colormap)
        plt.axis(extension)

        if only_image is True:
            plt.axis('off')
            return h

        plt.title(title, fontsize=16)

        if has_max is True:
            text_up = "{}".format(image.max())
            plt.text(self.x.max(),
                     self.z.max(),
                     text_up,
                     fontsize=14,
                     bbox=dict(edgecolor='white',
                               facecolor='white',
                               alpha=0.75))

            text_down = "{}".format(image.min())
            plt.text(self.x.max(),
                     self.z.min(),
                     text_down,
                     fontsize=14,
                     bbox=dict(edgecolor='white',
                               facecolor='white',
                               alpha=0.75))

        plt.xlabel("$z  (\mu m)$")
        plt.ylabel("$x  (\mu m)$")
        if colormap is not None:
            plt.colorbar(orientation='horizontal', fraction=0.046)
            h.set_clim(0, image.max())

        return h


def polarization_ellipse(self, pol_state=None, matrix=False):
    """returns A, B, theta, h polarization parameter of elipses

    Parameters:
        pol_state (None or (I, Q, U, V) ): Polarization state previously computed
        Matrix (bool): if True returns Matrix, else Scalar_field_X

    Returns:
        A, B, theta, h for Matrix=True
        CA, CB, Ctheta, Ch for Matrix=False
    """
    if pol_state is None:
        I, Q, U, V = self.polarization_states(matrix=True)
    else:
        I, Q, U, V = pol_state
        I = I.u
        Q = Q.u
        U = U.u
        V = V.u

    Ip = np.sqrt(Q**2 + U**2 + V**2)
    L = Q + 1.j * U

    A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
    B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
    theta = 0.5 * np.angle(L)
    h = np.sign(V)

    if matrix is True:
        return A, B, theta, h
    else:
        CA = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
        CB = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
        Ctheta = Scalar_field_XZ(x=self.x,
                                 z=self.z,
                                 wavelength=self.wavelength)
        Ch = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)

        CA.u = A
        CB.u = B
        Ctheta.u = theta
        Ch.u = h
        return (CA, CB, Ctheta, Ch)

    I = I.u
    Q = Q.u
    U = U.u
    V = V.u

    Ip = np.sqrt(Q**2 + U**2 + V**2)
    L = Q + 1.j * U
    A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
    B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
    theta = 0.5 * np.angle(L)
    h = np.sign(V)
    if matrix is True:
        return A, B, theta, h
    else:
        CA = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        CB = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        Ctheta = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        Ch = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        CA.u = A
        CB.u = B
        Ctheta.u = theta
        Ch.u = h
        return (CA, CB, Ctheta, Ch)

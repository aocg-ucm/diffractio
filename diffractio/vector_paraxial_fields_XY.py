# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_paraxial_field_XY class.

It is required also for generating masks and fields.
The main atributes are:
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.x - x positions of the field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.wavelength - wavdelength of the incident field. The field is monochromatic
    * self.Ex - x component of electric field. Equal size to x * y. complex field
    * self.Ey - y component of electric field. Equal size to x * y. complex field
    * self.X (numpy.array): equal size to x * y. complex field
    * self.Y (numpy.array): equal size to x * y. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for XY paraial vector fields*

*Definition of a scalar field*
    * add, substract fields
    * save, load data, clean, get, normalize
    * cut_resample
    * appy_mask

*Vector parameters*
    * polarization_states
    * polarization_ellipse

*Propagation*
    * RS - Rayleigh Sommerfeld

*Drawing functions*
    * draw: intensity, intensities, phases, fields, stokes, param_ellipse, ellipses

"""

from matplotlib import rcParams
from scipy.interpolate import RectBivariateSpline

from diffractio import degrees, eps, mm, np, plt
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.utils_common import load_data_common, save_data_common
from diffractio.utils_drawing import normalize_draw, reduce_matrix_size
from diffractio.utils_math import nearest


class Vector_paraxial_field_XY(object):
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
        self.x = x
        self.y = y
        self.wavelength = wavelength  # la longitud de onda
        self.X, self.Y = np.meshgrid(x, y)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)

    def __str__(self):
        """Represents data from class."""

        print('length x: ', self.x.shape, '- xmin: ', self.x[0], '- xmax: ',
              self.x[-1])
        print('length y: ', self.y.shape, '- ymin: ', self.y[0], '- ymax: ',
              self.y[-1])
        print('size Ex: ', self.Ex.shape)

    def __add__(self, other, kind='standard'):
        """adds two Vector_paraxial_field_XY. For example two light sources or two masks

        Parameters:
            other (Vector_paraxial_field_XY): 2nd field to add
            kind (str): instruction how to add the fields:
                - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1.
                - 'standard': not implemented yet

        Returns:
            Vector_paraxial_field_XY: `E3 = E1 + E2`
        """

        EM = Vector_paraxial_field_XY(self.x, self.y, self.wavelength)

        if kind == 'standard':
            EM.Ex = self.Ex + other.Ex
            EM.Ey = self.Ey + other.Ey

        return EM

    def __mul__(self, other):
        """Multiply two fields.

        Example:
            :math:`E_1(x)= E_0(x)*t(x)`

        Parameters:
            other (Vector_paraxial_field_XY): field to multiply

        Returns:
            Scalar_field_X: :math:`u_1(x)= u_0(x)*t(x)`
        """

        EM = Vector_paraxial_field_XY(self.x, self.y, self.wavelength)
        EM.Ex = self.Ex * other.Ex
        EM.Ey = self.Ey * other.Ey
        return EM

    def save_data(self, filename='', method='hickle', add_name=''):
        """Save data of Scalar_field_X class to a dictionary.

        Parameters:
            filename (str): filename
            method (str): 'savez', 'savez_compressed' 'hickle', 'matlab'.

        Returns:
            (bool): True if saving is performed, else False.
        """
        try:
            save_data_common(self, filename + add_name, method)
            return True
        except:
            return False

    def load_data(self, filename, method, verbose=False):
        """Load data from a file to a Scalar_field_X.

        Parameters:
            filename (str): filename
            method (str): 'savez', 'savez_compressed' 'hickle', 'matlab'.
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename, method, verbose)

        if verbose:
            print(dict0)
        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

    def clear(self):
        """simple - removes the field: self.E=0 """

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ex, dtype=complex)

    def get(self):
        """Takes the vector field and divide in Scalar_field_XY


        Returns:
            Scalar_field_XY: (Ex, Ey),
        """

        Ex = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        Ex.u = self.Ex
        Ey = Scalar_field_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        Ey.u = self.Ey

        return Ex, Ey

        return self.irradiance

    def apply_mask(self, u):
        """Multiply field by binary scalar mask
            for example: self.Ex = self.Ex * u.u

        Parameters:
           u (Scalar_mask_XY): mask
         """
        self.Ex = self.Ex * u.u
        self.Ey = self.Ey * u.u

    def RS(self, z=10 * mm, n=1, new_field=True):
        """Fast-Fourier-Transform  method for numerical integration of diffraction Rayleigh-Sommerfeld formula.

        From Applied Optics vol 45 num 6 pp. 1102-1110 (2006)

        `Thin Element Approximation` is considered for determining the field just after the mask: :math:`\mathbf{E}_{0}(\zeta,\eta)=t(\zeta,\eta)\mathbf{E}_{inc}(\zeta,\eta)`

        Is we have a field of size N*M, the result of propagation is also a field N*M. Nevertheless, there is a parameter `amplification` which allows us to determine the field in greater observation planes (jN)x(jM).

        One adventage of this approach is that it returns a quality parameter: if self.quality>1, propagation is right.

        Parameters:
            z (float): distance to observation plane.
                if z<0 inverse propagation is executed
            n (float): refraction index
            new_field (bool): if False the computation goes to self.u
                              if True a new instance is produced

            verbose (bool): if True it writes to shell. Not implemented yet

        Returns:
            if New_field is True: Scalar_field_X
            else None

        Todo:
            check amplification
            implement verbose
        """

        e0x, e0y = self.get()

        # estas son las components justo en la posicion pedida
        Ex = e0x.RS(z=z, n=n, new_field=True, kind='z')
        Ey = e0y.RS(z=z, n=n, new_field=True, kind='z')

        if new_field is True:
            EM = Vector_paraxial_field_XY(self.x, self.y, self.wavelength)
            EM.Ex = Ex.u
            EM.Ey = Ey.u
            return EM

        else:
            self.Ex = Ex.u
            self.Ey = Ey.u

    def polarization_states(self, matrix=False):
        """returns the Stokes parameters

        Parameters:
            Matrix (bool): if True returns Matrix, else Scalar_field_XY

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
            CI = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            CQ = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            CU = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            CV = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)

            CI.u = I
            CQ.u = Q
            CU.u = U
            CV.u = V

            return CI, CQ, CU, CV

    def polarization_ellipse(self, pol_state=None, matrix=False):
        """returns A, B, theta, h polarization parameter of elipses

        Parameters:
            pol_state (None or (I, Q, U, V) ): Polarization state previously computed
            Matrix (bool): if True returns Matrix, else Scalar_field_XY

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
            CA = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            CB = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            Ctheta = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)
            Ch = Scalar_field_XY(
                x=self.x, y=self.y, wavelength=self.wavelength)

            CA.u = A
            CB.u = B
            Ctheta.u = theta
            Ch.u = h
            return (CA, CB, Ctheta, Ch)

    def normalize(self):
        max_amplitude = np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2) + eps

        self.Ex = self.Ex / max_amplitude
        self.Ey = self.Ey / max_amplitude

    def cut_resample(self,
                     x_limits='',
                     y_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind=(3, 1)):
        """it cut the field to the range (x0,x1).
        if one of this x0,x1 positions is out of the self.x range it do nothing
        It is also valid for resampling the field, just write x0,x1 as
           the limits of self.x

        Parameters:
            x_limits (float,float): (x0,x1) starting and final points to cut
              if '' - takes the current limit x[0] and x[-1]
            y_limits (float,float): (y0,y1) - starting and final points to cut
              if '' - takes the current limit y[0] and y[-1]
            num_points (int): it resamples x, y and u
                [],'',0,None -> it leave the points as it is
            new_field (bool): it returns a new Scalar_field_XY
            interp_kind: numbers between 1 and 5
        """
        if x_limits == '':
            # used only for resampling
            x0 = self.x[0]
            x1 = self.x[-1]
        else:
            x0, x1 = x_limits

        if y_limits == '':
            # used only for resampling
            y0 = self.y[0]
            y1 = self.y[-1]
        else:
            y0, y1 = y_limits

        if x0 < self.x[0]:
            x0 = self.x[0]
        if x1 > self.x[-1]:
            x1 = self.x[-1]

        if y0 < self.y[0]:
            y0 = self.y[0]
        if y1 > self.y[-1]:
            y1 = self.y[-1]

        i_x0, _, _ = nearest(self.x, x0)
        i_x1, _, _ = nearest(self.x, x1)
        # new_num_points = i_x1 - i_x0
        i_y0, _, _ = nearest(self.y, y0)
        i_y1, _, _ = nearest(self.y, y1)

        kxu, kxn = interp_kind

        if num_points not in ([], '', 0, None):
            num_points_x, num_points_y = num_points
            x_new = np.linspace(x0, x1, num_points_x)
            y_new = np.linspace(y0, y1, num_points_y)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            f_interp_abs_x = RectBivariateSpline(
                self.x, self.y, np.abs(self.Ex), kx=kxu, ky=kxu, s=0)
            f_interp_phase_x = RectBivariateSpline(
                self.x, self.y, np.angle(self.Ex), kx=kxu, ky=kxu, s=0)

            f_interp_abs_y = RectBivariateSpline(
                self.x, self.y, np.abs(self.Ey), kx=kxu, ky=kxu, s=0)
            f_interp_phase_y = RectBivariateSpline(
                self.x, self.y, np.angle(self.Ey), kx=kxu, ky=kxu, s=0)

            Ex_new_abs = f_interp_abs_x(x_new, y_new)
            Ex_new_phase = f_interp_phase_x(x_new, y_new)
            Ex_new = Ex_new_abs * np.exp(1j * Ex_new_phase)

            Ey_new_abs = f_interp_abs_y(x_new, y_new)
            Ey_new_phase = f_interp_phase_y(x_new, y_new)
            Ey_new = Ey_new_abs * np.exp(1j * Ey_new_phase)

        else:
            i_s = slice(i_x0, i_x1)
            j_s = slice(i_y0, i_y1)
            x_new = self.x[i_s]
            y_new = self.y[j_s]
            X_new, Y_new = np.meshgrid(x_new, y_new)
            Ex_new = self.Ex[i_s, j_s]
            Ey_new = self.Ey[i_s, j_s]

        if new_field is False:
            self.x = x_new
            self.y = y_new
            self.Ex = Ex_new
            self.Ey = Ey_new
            self.X = X_new
            self.Y = Y_new
        elif new_field is True:
            field = Vector_paraxial_field_XY(
                x=x_new, y=y_new, wavelength=self.wavelength)
            field.Ex = Ex_new
            field.Ey = Ey_new
            return field

    def draw(self,
             kind='intensity',
             logarithm=False,
             normalize='',
             cut_value=1,
             numElipses=7,
             filename='',
             draw=True):
        """Draws electromagnetic field

        Parameters:
            kind (str):  'intensity', 'intensities', 'phases', field', 'stokes', 'param_ellipse'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            filename (str): if not '' stores drawing in file,
            cut_value (float): If not None, cuts the maximum intensity to this value
            numElipses (int): number of ellipses for parameters_ellipse
            amplification (float): amplification of ellipses
        """
        if draw is True:

            if kind == 'intensity':
                id_fig = self.__draw_intensity__(logarithm, normalize,
                                                 cut_value)
            if kind == 'intensities':
                id_fig = self.__draw_intensities__(logarithm, normalize,
                                                   cut_value)

            if kind == 'phases':
                id_fig = self.__draw_phases__()
            if kind == 'fields':
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value)

            if kind == 'stokes':
                id_fig = self.__draw_stokes__()

            if kind == 'param_ellipse':
                id_fig = self.__draw_param_ellipse__()

            if not filename == '':
                plt.savefig(
                    filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

            return id_fig

    def __draw_intensity__(self, logarithm=False, normalize='', cut_value=''):
        """Draws the intensity

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)

        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)

        intensity = np.abs(Ex_r)**2 + np.abs(Ey_r)**2
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.figure()
        h1 = __draw1__(self, intensity, "gist_heat", "$I$")
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return h1

    def __draw_intensities__(self, logarithm=False, normalize='',
                             cut_value=''):
        """internal funcion: draws intensity X,Y,Z frames for E and H

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        color_amplitude = "gist_heat"
        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)

        intensity_max = max((np.abs(Ex_r)**2).max(), (np.abs(Ey_r)**2).max())

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, ty))

        plt.subplot(1, 2, 1)
        intensity = np.abs(Ex_r)**2
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)
        h1 = __draw1__(self, intensity, color_amplitude, "$I_x$")
        plt.clim(0, intensity_max)
        plt.subplot(1, 2, 2)
        intensity = np.abs(Ey_r)**2
        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)
        h2 = __draw1__(self, intensity, color_amplitude, "$I_y$")
        plt.clim(0, intensity_max)

        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()

        return h1, h2

    def __draw_phases__(self):
        """internal funcion: draws intensity X,Y,Z frames for E and H

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        color_phase = "RdBu"
        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)
        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, ty))

        plt.subplot(1, 2, 1)
        phase = np.angle(Ex_r)
        h1 = __draw1__(self, phase / degrees, color_phase, "$E_x$")

        plt.subplot(1, 2, 2)
        phase = np.angle(Ey_r)
        h2 = __draw1__(self, phase / degrees, color_phase, "$E_y$")
        plt.clim(-180, 180)

        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()

        return h1, h2

    def __draw_fields__(self, logarithm=False, normalize='', cut_value=''):
        """__internal__: draws amplitude and phase in 2x3 drawing

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        color_amplitude = "gist_heat"
        color_phase = "RdBu"

        Ex_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ex)
        Ey_r = reduce_matrix_size(self.reduce_matrix, self.x, self.y, self.Ey)

        amplitude_max = max(np.abs(Ex_r).max(), np.abs(Ey_r).max())

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        plt.subplot(2, 2, 1)
        amplitude = np.abs(self.Ex)
        h1 = __draw1__(self, amplitude, color_amplitude, "$A_x$")
        plt.clim(0, amplitude_max)

        plt.subplot(2, 2, 2)
        amplitude = np.abs(self.Ey)
        h2 = __draw1__(self, amplitude, color_amplitude, "$A_y$")
        plt.clim(0, amplitude_max)

        plt.subplot(2, 2, 3)
        phase = np.angle(self.Ex)
        h1 = __draw1__(self, phase / degrees, color_phase, "$\phi_x$")
        plt.clim(-180, 180)

        plt.subplot(2, 2, 4)
        phase = np.angle(self.Ey)
        h2 = __draw1__(self, phase / degrees, color_phase, "$\phi_y$")
        plt.clim(-180, 180)

        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return (h1, h2)

    def __draw_stokes__(self):
        """__internal__: computes and draws CI, CQ, CU, CV parameters
        """

        tx, ty = rcParams['figure.figsize']

        S0, S1, S2, S3 = self.polarization_states(matrix=True)

        intensity_max = S0.max()

        plt.figure(figsize=(2 * tx, 2 * ty))
        plt.subplot(2, 2, 1)
        h1 = __draw1__(self, S0, "gist_heat", "$S_0$")
        plt.clim(0, intensity_max)

        plt.subplot(2, 2, 2)
        h2 = __draw1__(self, S1, "RdBu", "$S_1$")
        plt.clim(-intensity_max, intensity_max)

        plt.subplot(2, 2, 3)
        h3 = __draw1__(self, S2, "RdBu", "$S_2$")
        plt.clim(-intensity_max, intensity_max)

        plt.subplot(2, 2, 4)
        h4 = __draw1__(self, S3, "RdBu", "$S_3$")
        plt.clim(-intensity_max, intensity_max)

        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_param_ellipse__(self):
        """__internal__: computes and draws polariations ellipses
        """
        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.y, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.y, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.y, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.y, h)

        tx, ty = rcParams['figure.figsize']

        plt.figure(figsize=(2 * tx, 2 * ty))

        plt.subplot(2, 2, 1)
        h1 = __draw1__(self, A, "gist_heat", "$A$")
        plt.subplot(2, 2, 2)
        h2 = __draw1__(self, B, "gist_heat", "$B$")
        plt.subplot(2, 2, 3)
        h3 = __draw1__(self, theta / degrees, "RdBu", "$\phi$")
        plt.clim(-180, 180)
        plt.subplot(2, 2, 4)
        h4 = __draw1__(self, h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_ellipses__(self, numElipses=5, amplification=7):
        """__internal__: draw ellipses

        Parameters:
            numElipses (int): number of ellipses for parameters_ellipse
            amplification (float): amplification of ellipses
        """
        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        x0 = np.linspace(self.x[0], self.x[-1], numElipses)
        y0 = np.linspace(self.y[0], self.y[-1], numElipses)

        for xi in x0:
            [i, tmp, tmp] = nearest(self.x, xi)
            for yj in y0:
                [j, tmp, tmp] = nearest(self.y, yj)
                x, y = _compute1Elipse__(xi, yj, A[i, j], B[i, j], theta[i, j],
                                         1, amplification)
                if h[i, j] == 1:
                    plt.plot(x, y, 'b')
                else:
                    plt.plot(x, y, 'r')

        plt.xlim(self.x[0], self.x[-1])
        plt.ylim(self.y[0], self.y[-1])
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()


def __draw1__(hdl, image, colormap, title='', has_max=False):
    """Draws image

    Parameters:
        image (numpy.array): array with drawing
        colormap (str): colormap
        title (str): title of drawing
    """
    extension = [hdl.x.min(), hdl.x.max(), hdl.y.min(), hdl.y.max()]
    h = plt.imshow(
        image,
        interpolation='bilinear',
        aspect='auto',
        origin='lower',
        extent=extension)

    title = title
    plt.title(title, fontsize=16)

    if has_max is True:
        text_up = "{}".format(image.max())
        plt.text(
            hdl.x.max(),
            hdl.y.max(),
            text_up,
            fontsize=14,
            bbox=dict(edgecolor='white', facecolor='white', alpha=0.75))

        text_down = "{}".format(image.min())
        plt.text(
            hdl.x.max(),
            hdl.y.min(),
            text_down,
            fontsize=14,
            bbox=dict(edgecolor='white', facecolor='white', alpha=0.75))

    plt.xlabel("$x  (\mu m)$")
    plt.ylabel("$y  (\mu m)$")
    plt.colorbar(orientation='horizontal', shrink=0.66)

    h.set_cmap(colormap)
    plt.axis('scaled')
    plt.axis(extension)

    return h


def _compute1Elipse__(x0, y0, A, B, theta, h=0, amplification=1):
    """computes polarization ellipse for drawing

    Parameters:
        x0 (float): position x of ellipse
        y0 (float): position y of ellipse
        A (float): axis 1 of ellipse
        B (float): axis 2 of ellipse
        theta (float): angle of ellipse
        h (float): to remove
        amplification (float): increase of size of ellipse

    Todo:
        remove hs
    """
    # esto es para verlo más grande
    A = A * amplification
    B = B * amplification

    cos = np.cos
    sin = np.sin

    fi = np.linspace(0, 2 * np.pi, 64)
    cf = cos(fi - theta)
    sf = sin(fi - theta)

    r = 1 / np.sqrt(np.abs(cf / (A + eps)**2 + sf**2 / (B + eps)**2))

    x = r * cos(fi) + x0
    y = r * sin(fi) + y0

    return x, y

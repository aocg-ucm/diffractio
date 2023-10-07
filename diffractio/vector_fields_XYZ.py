# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Vector_field_XYZ class. It is required also for generating masks and fields.
The main atributes are:
    * self.Ex - x component of electric field
    * self.Ey - y component of electric field
    * self.Ez - z component of electric field
    * self.wavelength - wavelength of the incident field. The field is monocromatic
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.z - z positions of the field
    * self.X (numpy.array): equal size to x * y * z. complex field
    * self.Y (numpy.array): equal size to x * y * z. complex field
    * self.Z (numpy.array): equal size to x * y * z. complex field
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date when performed


The magnitude is related to microns: `micron = 1.`

*Class for XY vector fields*

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
import copy

from . import degrees, eps, mm, np, plt
from .config import CONF_DRAWING
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XY import Scalar_field_XY
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_fields_XYZ import Scalar_field_XYZ
from .scalar_masks_XY import Scalar_mask_XY
from .vector_fields_XZ import Vector_field_XZ
from .utils_common import load_data_common, save_data_common, get_date
from .utils_math import ndgrid, nearest
from .utils_optics import normalize_field

percentage_intensity = CONF_DRAWING['percentage_intensity']


class Vector_field_XYZ(object):
    """Class for vectorial fields.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        z (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.y (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly 2**n.
        self.wavelength (float): wavelength of the incident field.
        self.Ex (numpy.array): Electric_x field
        self.Ey (numpy.array): Electric_y field
        self.Ez (numpy.array): Electric_z field
    """

    def __init__(self, x, y, z, wavelength, info=''):
        self.x = x
        self.y = y
        self.z = z
        self.wavelength = wavelength  # la longitud de onda
        self.X, self.Y, self.Z = ndgrid(self.x, self.y, self.z)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.reduce_matrix = 'standard'  # 'None, 'standard', (5,5)
        self.type = 'Vector_field_XYZ'
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING


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
            " - ymin:       {:2.2f} um,  ymay:      {:2.2f} um,  Dy:   {:2.2f} um"
            .format(self.y[0], self.y[-1], self.y[1] - self.y[0]))
        print(
            " - zmin:       {:2.2f} um,  zmaz:      {:2.2f} um,  Dz:   {:2.2f} um"
            .format(self.z[0], self.z[-1], self.z[1] - self.z[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""

    def __add__(self, other, kind='standard'):
        """adds two Vector_field_XY. For example two light sources or two masks

        Parameters:
            other (Vector_field_XY): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_XY: `E3 = E1 + E2`
        """

        EM = Vector_field_XYZ(self.x, self.y, self.z, self.wavelength)

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
        """Load data from a file to a Vector_field_XY.
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
        """simple - removes the field: self.E=0 """

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ex, dtype=complex)
        self.Ez = np.zeros_like(self.Ex, dtype=complex)

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def get(self, kind='fields', is_matrix=True):
        """Takes the vector field and divide in Scalar_field_XYZ

        Parameters:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Scalar_field_XYZ: (Ex, Ey),
        """

        Ex_r = self.Ex
        Ey_r = self.Ey
        Ez_r = self.Ez

        if kind == 'fields':
            if is_matrix:
                return self.Ex, self.Ey, self.Ez

            else:
                Ex = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
                Ex.u = Ex_r
                Ey = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
                Ey.u = Ey_r
                Ez = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
                Ez.u = Ez_r
                return Ex, Ey, Ez

        elif kind == 'intensity':
            intensity = np.abs(Ex_r)**2 + np.abs(Ey_r)**2 + np.abs(Ez_r)**2

            if is_matrix:
                return intensity

            else:
                Intensity = Scalar_field_XYZ(x=self.x,
                                             y=self.y,
                                             z=self.z,
                                             wavelength=self.wavelength)
                Intensity.u = np.sqrt(intensity)

                return Intensity

        elif kind == 'intensities':
            intensity_x = np.abs(Ex_r)**2
            intensity_y = np.abs(Ey_r)**2
            intensity_z = np.abs(Ez_r)**2
            return intensity_x, intensity_y, intensity_z

        elif kind == 'phases':
            phase_x = np.angle(Ex_r)
            phase_y = np.angle(Ey_r)
            phase_z = np.angle(Ez_r)

            if is_matrix:
                return phase_x, phase_y, phase_z
            else:
                Ex = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
                Ex.u = np.exp(1j * phase_x)
                Ey = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
                Ey.u = np.exp(1j * phase_y)
                Ez = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
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

    def intensity(self):
        """"Returns intensity.
        """
        intensity = np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(
            self.Ez)**2

        return intensity

    def polarization_states(self, matrix=False):
        """returns the Stokes parameters

        Parameters:
            Matrix (bool): if True returns Matrix, else Scalar_field_XYZ

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
            CI = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
                                  z=self.z,
                                  wavelength=self.wavelength)
            CQ = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
                                  z=self.z,
                                  wavelength=self.wavelength)
            CU = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
                                  z=self.z,
                                  wavelength=self.wavelength)
            CV = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
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
            Matrix (bool): if True returns Matrix, else Scalar_field_XYZ

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
            CA = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
                                  z=self.z,
                                  wavelength=self.wavelength)
            CB = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
                                  z=self.z,
                                  wavelength=self.wavelength)
            Ctheta = Scalar_field_XYZ(x=self.x,
                                      y=self.y,
                                      z=self.z,
                                      wavelength=self.wavelength)
            Ch = Scalar_field_XYZ(x=self.x,
                                  y=self.y,
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

    def to_Vector_field_XY(self,
                           iz0=None,
                           z0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Scalar_field_XY. Only one of the first two variables (iz0,z0) should be used

        Parameters:
            iz0 (int): position i of z data in array
            z0 (float): position z to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        TODO: Simplify and change variable name clase
        """
        if is_class is True:
            field_output = Scalar_field_XY(x=self.x,
                                           y=self.y,
                                           wavelength=self.wavelength)
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
            field_output.u = np.squeeze(self.u[:, :, iz])
            return field_output

        if matrix is True:
            if iz0 is None:
                iz, tmp1, tmp2 = nearest(self.z, z0)
            else:
                iz = iz0
            return np.squeeze(self.u[:, :, iz])

    def to_Vector_field_XZ(self,
                           iy0=None,
                           y0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Vector_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Parameters:
            iy0 (int): position i of y data in array
            y0 (float): position y to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        if is_class is True:
            field_output = Vector_field_XZ(x=self.x,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if iy0 is None:
                iy, tmp1, tmp2 = nearest(self.y, y0)
            else:
                iy = iy0
            field_output.Ex = np.squeeze(self.Ex[:, iy, :])
            field_output.Ey = np.squeeze(self.Ey[:, iy, :])
            field_output.Ez = np.squeeze(self.Ez[:, iy, :])
            return field_output

        if matrix is True:
            if iy0 is None:
                iy, tmp1, tmp2 = nearest(self.y, y0)
            else:
                iy = iy0
            return np.squeeze(self.Ex[:, iy, :]), np.squeeze(self.Ey[:, iy, :]), np.squeeze(self.Ez[:, iy, :])

    def to_Vector_field_YZ(self,
                           ix0=None,
                           x0=None,
                           is_class=True,
                           matrix=False):
        """pass results to Vector_field_XZ. Only one of the first two variables (iy0,y0) should be used

        Parameters:
            ix0 (int): position i of x data in array
            x0 (float): position x to extract
            class (bool): If True it returns a class
            matrix (bool): If True it returns a matrix

        """
        if is_class is True:
            field_output = Scalar_field_XZ(x=self.y,
                                           z=self.z,
                                           wavelength=self.wavelength)
            if ix0 is None:
                ix, tmp1, tmp2 = nearest(self.x, x0)
            else:
                iy = ix0
            field_output.Ex = np.squeeze(self.Ex[ix, :, :])
            field_output.Ey = np.squeeze(self.Ey[ix, :, :])
            field_output.Ez = np.squeeze(self.Ez[ix, :, :])            
            return field_output

        if matrix is True:
            if ix0 is None:
                ix, _, _ = nearest(self.x, x0)
            else:
                ix = ix0
            return np.squeeze(self.Ex[ix, :, :]), np.squeeze(self.Ey[ix, :, :]), np.squeeze(self.Ez[ix, :, :])

    def to_Z(self,
             kind='amplitude',
             x0=None,
             y0=None,
             has_draw=True,
             z_scale='mm'):
        """pass results to u(z). Only one of the first two variables (iy0,y0) and (ix0,x0) should be used.

        Parameters:
            kind (str): 'amplitude', 'intensity', 'phase'
            x0 (float): position x to extract
            y0 (float): position y to extract
            has_draw (bool): draw the field
            z_scale (str): 'mm', 'um'

        Returns:
            z (numpy.array): array with z
            field (numpy.array): amplitude, intensity or phase of the field
        """
        ix, _, _ = nearest(self.y, y0)
        iy, _, _ = nearest(self.x, x0)

        u = np.squeeze(self.u[ix, iy, :])

        if kind == 'amplitude':
            y = np.abs(u)
        elif kind == 'intensity':
            y = np.abs(u)**2
        elif kind == 'phase':
            y = np.angle(u)

        if has_draw is True:
            if z_scale == 'mm':
                plt.plot(self.z / mm, y, 'k', lw=2)
                plt.xlabel('$z\,(mm)$')
                plt.xlim(left=self.z[0] / mm, right=self.z[-1] / mm)

            elif z_scale == 'um':
                plt.plot(self.z, y, 'k', lw=2)
                plt.xlabel('$z\,(\mu m)$')
                plt.xlim(left=self.z[0], right=self.z[-1])

            plt.ylabel(kind)

    def draw_XY(self,
                z0=5 * mm,
                kind='intensity',
                logarithm=0,
                normalize='maximum',
                title='',
                filename='',
                cut_value='',
                has_colorbar='False',
                reduce_matrix=''):
        """ longitudinal profile XY at a given z value

        Parameters:
            z0 (float): value of z for interpolation
            kind (str): type of drawing: 'amplitude', 'intensity', 'phase', ' 'field', 'real_field', 'contour'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'area', 'intensity'
            title (str): title for the drawing
            filename (str): if not '' stores drawing in file,
            cut_value (float): if provided, maximum value to show
            has_colorbar (bool): if True draws the colorbar
            reduce_matrix ()
        """

        ufield = self.to_Vector_field_XY(iz0=None,
                                         z0=z0,
                                         is_class=True,
                                         matrix=False)
        ufield.draw(kind=kind,
                    logarithm=logarithm,
                    normalize=normalize,
                    title=title,
                    filename=filename,
                    cut_value=cut_value,
                    has_colorbar=has_colorbar,
                    reduce_matrix=reduce_matrix)

    def draw_XZ(self,
                kind='intensity',
                y0=0 * mm,
                logarithm=0,
                normalize='',
                draw_borders=False,
                filename='',
                **kwargs):
        """Longitudinal profile XZ at a given x0 value.

        Parameters:
            y0 (float): value of y for interpolation
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Vector_field_XZ(y0=y0)
        h1 = ufield.draw(kind, logarithm, normalize, draw_borders, filename,
                         **kwargs)
        # intensity = np.abs(ufield.u)**2

        # if logarithm == 1:
        #     intensity = np.log(intensity + 1)

        # if normalize == 'maximum':
        #     intensity = intensity / intensity.max()
        # if normalize == 'area':
        #     area = (self.x[-1] - self.x[0]) * (self.z[-1] - self.z[0])
        #     intensity = intensity / area
        # if normalize == 'intensity':
        #     intensity = intensity / (intensity.sum() / len(intensity))

        # h1 = plt.imshow(intensity,
        #                 interpolation='bilinear',
        #                 aspect='auto',
        #                 origin='lower',
        #                 extent=[
        #                     self.z[0] / 1000, self.z[-1] / 1000, self.y[0],
        #                     self.y[-1]
        #                 ])
        # plt.xlabel('z (mm)', fontsize=16)
        # plt.ylabel('x $(um)$', fontsize=16)
        # plt.title('intensity XZ', fontsize=20)
        # h1.set_cmap(
        #     self.CONF_DRAWING['color_intensity'])  # OrRd # Reds_r gist_heat
        # plt.colorbar()

        # # -----------------     no functiona de momento -----------------
        # if draw_borders is True:
        #     x_surface, y_surface, z_surface, x_draw_intensity, y_draw_intensity, z_draw_intensity = self.surface_detection(
        #     )
        #     plt.plot(y_draw_intensity, z_draw_intensity, 'w.', ms=2)

        # if not filename == '':
        #     plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

        return h1

    def draw_YZ(self,
                x0=0 * mm,
                logarithm=0,
                normalize='',
                draw_borders=False,
                filename=''):
        """Longitudinal profile YZ at a given x0 value.

        Parameters:
            x0 (float): value of x for interpolation
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (str):  False, 'maximum', 'intensity', 'area'
            draw_borders (bool): check
            filename (str): filename to save
        """

        plt.figure()
        ufield = self.to_Vector_field_YZ(x0=x0)
        intensity = np.abs(ufield.u)**2

        if logarithm == 1:
            intensity = np.log(intensity + 1)

        if normalize == 'maximum':
            intensity = intensity / intensity.max()
        if normalize == 'area':
            area = (self.y[-1] - self.y[0]) * (self.z[-1] - self.z[0])
            intensity = intensity / area
        if normalize == 'intensity':
            intensity = intensity / (intensity.sum() / len(intensity))

        h1 = plt.imshow(intensity,
                        interpolation='bilinear',
                        aspect='auto',
                        origin='lower',
                        extent=[
                            self.z[0] / 1000, self.z[-1] / 1000, self.y[0],
                            self.y[-1]
                        ])
        plt.xlabel('z (mm)', fontsize=16)
        plt.ylabel('y $(um)$', fontsize=16)
        plt.title('intensity YZ', fontsize=20)
        h1.set_cmap(
            self.CONF_DRAWING['color_intensity'])  # OrRd # Reds_r gist_heat
        plt.colorbar()

        # -----------------     no functiona de momento -----------------
        if draw_borders is True:
            x_surface, y_surface, z_surface, x_draw_intensity, y_draw_intensity, z_draw_intensity = self.surface_detection(
            )
            plt.plot(y_draw_intensity, z_draw_intensity, 'w.', ms=2)

        if not filename == '':
            plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

        return h1


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

    TODO: Remove hs
    """
    # esto es para verlo más grande
    A = A * amplification
    B = B * amplification

    fi = np.linspace(0, 2 * np.pi, 64)
    cf = np.cos(fi - theta)
    sf = np.sin(fi - theta)

    r = 1 / np.sqrt(np.abs(cf / (A + eps)**2 + sf**2 / (B + eps)**2))

    x = r * np.cos(fi) + x0
    y = r * np.sin(fi) + y0

    return x, y
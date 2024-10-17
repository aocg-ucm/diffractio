# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        vector_fields_XZ.py
# Purpose:     Class for handling vector fields in the XZ plane
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


# flake8: noqa

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
import time
from scipy.interpolate import RectBivariateSpline


from .__init__ import degrees, eps, mm, np, plt
from .config import bool_raise_exception, CONF_DRAWING, Draw_Vector_XZ_Options
from .utils_typing import npt, Any, NDArray, NDArrayFloat, NDArrayComplex
from .utils_common import get_date, load_data_common, save_data_common, check_none
from .utils_drawing import normalize_draw, reduce_matrix_size
from .utils_math import get_k, nearest
from .utils_optics import normalize_field, fresnel_equations_kx
from .scalar_fields_X import Scalar_field_X
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_XZ import Scalar_mask_XZ
from .vector_fields_X import Vector_field_X

from py_pol.jones_vector import Jones_vector

from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft, ifftshift

percentage_intensity_config = CONF_DRAWING['percentage_intensity']


class Vector_field_XZ(Scalar_mask_XZ):
    """Class for vectorial fields.

    Args:
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

    def __init__(self, x: NDArrayFloat | None = None, z: NDArrayFloat | None = None,
                 wavelength: float | None = None, n_background: float = 1., info: str = ""):
        self.x = x
        self.z = z
        self.wavelength = wavelength  # la longitud de onda
        self.n_background = n_background

        self.X, self.Z = np.meshgrid(x, z)

        self.Ex = np.zeros_like(self.X, dtype=complex)
        self.Ey = np.zeros_like(self.X, dtype=complex)
        self.Ez = np.zeros_like(self.X, dtype=complex)

        self.Hx = None
        self.Hy = None
        self.Hz = None

        self.n = n_background * np.ones(np.shape(self.X), dtype=complex)
        self.borders = None  # borders at refractive index

        self.Ex0 = np.zeros_like(self.x)
        self.Ey0 = np.zeros_like(self.x)

        self.reduce_matrix = "standard"  # 'None, 'standard', (5,5)
        self.type = "Vector_field_XZ"
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING


    @check_none('x','z','Ex','Ey',raise_exception=bool_raise_exception)
    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print("{}\n - x:  {},   z:  {},   Ex:  {}".format(
            self.type, self.x.shape, self.z.shape, self.Ex.shape))

        print(
            " - xmin:       {:2.2f} um,  xmax:      {:2.2f} um,  Dx:   {:2.2f} um".format(
                self.x[0], self.x[-1], self.x[1] - self.x[0]
            )
        )
        print(
            " - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um".format(
                self.z[0], self.z[-1], self.x[1] - self.z[0]
            )
        )
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(Imin, Imax))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        print(" - info:       {}".format(self.info))

        return ""


    @check_none('x','z','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __add__(self, other):
        """adds two Vector_field_X. For example two light sources or two masks

        Args:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_X: `E3 = E1 + E2`
        """

        EM = Vector_field_XZ(self.x, self.z, self.wavelength)

        EM.Ex = self.Ex + other.Ex
        EM.Ey = self.Ey + other.Ey
        EM.Ez = self.Ez + other.Ez

        return EM


    def save_data(self, filename: str, add_name: str = "",
                  description: str = "", verbose: bool = False):
        """Common save data function to be used in all the modules.
        The methods included are: npz, matlab


        Args:
            filename (str): filename
            add_name= (str): sufix to the name, if 'date' includes a date
            description (str): text to be stored in the dictionary to save.
            verbose (bool): If verbose prints filename.

        Returns:
            (str): filename. If False, file could not be saved.
        """
        try:
            final_filename = save_data_common(
                self, filename, add_name, description, verbose
            )
            return final_filename
        except:
            return False


    def load_data(self, filename: str, verbose: bool = False):
        """Load data from a file to a Vector_field_X.
            The methods included are: npz, matlab

        Args:
            filename (str): filename
            verbose (bool): shows data process by screen
        """
        dict0 = load_data_common(self, filename)

        if dict0 is not None:
            if isinstance(dict0, dict):
                self.__dict__ = dict0
            else:
                raise Exception("no dictionary in load_data")

        if verbose is True:
            print(dict0.keys())


    @check_none('Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def clear_field(self):
        """Removes the fields Ex, Ey, Ez"""

        self.Ex = np.zeros_like(self.Ex, dtype=complex)
        self.Ey = np.zeros_like(self.Ey, dtype=complex)
        self.Ez = np.zeros_like(self.Ez, dtype=complex)


    def duplicate(self, clear: bool = False):
        """Duplicates the instance"""
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field


    @check_none('x','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def incident_field(self, E0: Vector_field_X  | None = None, u0: Scalar_field_X  | None = None, 
                       j0: Jones_vector  | None = None, z0: float | None = None):
        """Includes the incident field in Vector_field_XZ. 
        
        It can be performed using a Vector_field_X E0 or a Scalar_field_X u0 + Jones_vector j0.

        Args:
            E0 (Vector_field_X | None): Vector field of the incident field.
            u0 (Scalar_field_x | None): Scalar field of the incident field.
            j0 (py_pol.Jones_vector | None): Jones vector of the incident field.
            z0 (float | None): position of the incident field. if None, the field is at the beginning.
        """

        if np.logical_and.reduce((E0 is None, u0 is not None, j0 is not None)):
            E0 = Vector_field_X(self.x, self.wavelength, self.n_background)
            E0.Ex = u0.u * j0.M[0]
            E0.Ey = u0.u * j0.M[1]

        if z0 in (None, '', []):
            self.Ex0 = E0.Ex
            self.Ey0 = E0.Ey

            self.Ex[0, :] = self.Ex[0, :] + E0.Ex
            self.Ey[0, :] = self.Ey[0, :] + E0.Ey
        else:
            iz, _, _ = nearest(self.z, z0)
            self.Ex[iz, :] = self.Ex[iz, :] + E0.Ex
            self.Ey[iz, :] = self.Ey[iz, :] + E0.Ey


    @check_none('x','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def final_field(self):
        """Returns the final field as a Vector_field_X."""

        EH_final = Vector_field_X(x=self.x,
                                  wavelength=self.wavelength,
                                  n_background=self.n_background,
                                  info="from final_field at z0= {} um".format(
                                      self.z[-1]))
        EH_final.Ex = self.Ex[-1, :]
        EH_final.Ey = self.Ey[-1, :]
        EH_final.Ez = self.Ez[-1, :]
        EH_final.Hx = self.Hx[-1, :]
        EH_final.Hy = self.Hy[-1, :]
        EH_final.Hz = self.Hz[-1, :]
        return EH_final


    def refractive_index_from_scalarXZ(self, u_xz: Scalar_mask_XZ):
        """
        refractive_index_from_scalarXZ. Gets the refractive index from a Scalar field and passes to a vector field.
        
        Obviously, the refractive index is isotropic.

        Args:
            self (Vector_field_XZ): Vector_field_XZ
            u_xz (Scalar_mask_XZ): Scalar_mask_XZ
        """
        self.n = u_xz.n
        
        edges = self.surface_detection( min_incr = 0.1, reduce_matrix = 'standard', has_draw = False)
               
        self.borders = edges           
        return edges
        

    @check_none('x','z','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def get(self, kind: str = "fields", is_matrix: bool =True):
        """Takes the vector field and divide in Scalar_field_X.

        Args:
            kind (str): 'fields', 'intensity', 'intensities', 'phases', 'stokes', 'params_ellipse'

        Returns:
            Vector_field_X: (Ex, Ey, Ez),
        """

        self.Ex = self.Ex
        self.Ey = self.Ey
        self.Ez = self.Ez

        if kind == "fields":
            if is_matrix:
                return self.Ex, self.Ey, self.Ez

            else:
                Ex = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ex.u = self.Ex
                Ey = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ey.u = self.Ey
                Ez = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ez.u = self.Ez
                return Ex, Ey, Ez

        elif kind == "intensity":
            intensity = (
                np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2
            )

            if is_matrix:
                return intensity

            else:
                Intensity = Scalar_field_XZ(
                    x=self.x, z=self.z, wavelength=self.wavelength
                )
                Intensity.u = np.sqrt(intensity)

                return Intensity

        elif kind == "intensities":
            intensity_x = np.abs(self.Ex) ** 2
            intensity_y = np.abs(self.Ey) ** 2
            intensity_z = np.abs(self.Ez) ** 2
            return intensity_x, intensity_y, intensity_z

        elif kind == "phases":
            phase_x = np.angle(self.Ex)
            phase_y = np.angle(self.Ey)
            phase_z = np.angle(self.Ez)

            if is_matrix:
                return phase_x, phase_y, phase_z
            else:
                Ex = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ex.u = np.exp(1j * phase_x)
                Ey = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ey.u = np.exp(1j * phase_y)
                Ez = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
                Ez.u = np.exp(1j * phase_z)
                return Ex, Ey, Ez

        elif kind == "stokes":
            # S0, S1, S2, S3
            return self.polarization_states(matrix=True)

        elif kind == "params_ellipse":
            # A, B, theta, h
            return self.polarization_ellipse(pol_state=None, matrix=True)

        else:
            print("The parameter '{}'' in .get(kind='') is wrong".format(kind))

    @check_none('x','z','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def apply_mask(self, u, new_field: bool = False):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Args:
            u (Scalar_mask_X): mask
        """
        if new_field == False:
            self.Ex = self.Ex * u.u
            self.Ey = self.Ey * u.u
            self.Ez = self.Ez * u.u
        else:
            E_new = self.duplicate()
            E_new.Ex = self.Ex * u.u
            E_new.Ey = self.Ey * u.u
            E_new.Ez = self.Ez * u.u
            return E_new

    @check_none('x','z',raise_exception=bool_raise_exception)
    def FP_WPM(self, has_edges: bool = True, pow_edge: int = 80, matrix: bool = False, has_H=True, verbose: bool = False
    ):
        """
        WPM Method. 'schmidt methodTrue is very fast, only needs discrete number of refractive indexes'


        Args:
            has_edges (bool): If True absorbing edges are used.
            pow_edge (float): If has_edges, power of the supergaussian
            matrix (bool): if True returns a matrix else
            has_H (bool): If True, it returns magnetic field H.
            verbose (bool): If True prints information

        References:

            1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

            2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.

        """

        k0 = 2 * np.pi / self.wavelength

        x = self.x
        z = self.z

        dx = x[1] - x[0]
        dz = z[1] - z[0]

        self.Ex[0,:] = self.Ex0
        self.Ey[0,:] = self.Ey0

        if has_H:
            self.Hx = np.zeros_like(self.Ex)
            self.Hy = np.zeros_like(self.Ex)
            self.Hz = np.zeros_like(self.Ex)

        kx = get_k(x, flavour="+")

        if has_edges is False:
            has_filter = np.zeros_like(self.z)
        elif has_edges is True:
            has_filter = np.ones_like(self.z)
        elif isinstance(has_edges, (int, float)):
            has_filter = np.zeros_like(self.z)
            iz, _, _ = nearest(self.z, has_edges)
            has_filter[iz:] = 1
        else:
            has_filter = has_edges

        width_edge = 0.95*(self.x[-1]-self.x[0])/2
        x_center = (self.x[-1] + self.x[0])/2

        filter_function = np.exp(
            -((np.abs(self.x - x_center) / width_edge) ** pow_edge)
        )

        t1 = time.time_ns()

        num_steps = len(self.z)

        for j in range(1, num_steps):

            if has_filter[j] == 0:
                filter_edge = 1
            else:
                filter_edge = filter_function

            E_step, H_step = FP_WPM_schmidt_kernel(
                self.Ex[j - 1, :],
                self.Ey[j - 1, :],
                self.n[j - 1, :],
                self.n[j, :],
                k0,
                kx,
                self.wavelength,
                dz,
            ) * filter_edge

            self.Ex[j, :] = self.Ex[j, :] + E_step[0] * filter_edge
            self.Ey[j, :] = self.Ey[j, :] + E_step[1] * filter_edge
            self.Ez[j, :] = E_step[2] * filter_edge

            if has_H:
                self.Hx[j, :] = H_step[0] * filter_edge
                self.Hy[j, :] = H_step[1] * filter_edge
                self.Hz[j, :] = H_step[2] * filter_edge



        # at the initial point the Ez field is not computed.
        self.Ex[0,:] = self.Ex[1,:]
        self.Ey[0,:] = self.Ey[1,:]
        self.Ez[0,:] = self.Ez[1,:]
        
        if has_H:
            self.Hx[0,:] = self.Hx[1,:]
            self.Hy[0,:] = self.Hy[1,:]
            self.Hz[0,:] = self.Hz[1,:]

        t2 = time.time_ns()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    (t2 - t1) / 1e9, (t2 - t1) / len(self.z) / 1e6
                )
            )

        if matrix is True:
            return (self.Ex, self.Ey, self.Ez), (self.Hx, self.Hy, self.Hz)


    @check_none('Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def intensity(self):
        """ "Returns intensity."""
        intensity = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2

        return intensity


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def Poynting_vector(self, has_draw: bool = True, draw_borders: bool = True,  scale: str = ''):
        "Instantaneous Poynting Vector"

        Sx = self.Ey * self.Hz - self.Ez * self.Hy
        Sy = self.Ez * self.Hx - self.Ex * self.Hz
        Sz = self.Ex * self.Hy - self.Ey * self.Hx

        S_max = np.real(np.max((Sx, Sy, Sz)))
        S_min = np.real(np.min((Sx, Sy, Sz)))
        S_lim = np.max((abs(S_max), np.abs(S_min)))

        if has_draw:
            tx, ty = rcParams["figure.figsize"]

            dims = np.shape(self.Ex)
            num_dims = len(dims)
            if num_dims == 1:
                z0 = self.z

                plt.figure(figsize=(1 * tx, 3 * ty))
                plt.subplot(1, 3, 1)
                plt.plot(z0, np.real(Sx))
                plt.ylim(-S_lim, S_lim)
                plt.title("$S_x$")

                plt.subplot(1, 3, 2)
                plt.plot(z0, np.real(Sy))
                plt.ylim(-S_lim, S_lim)
                plt.title("$S_y$")

                plt.subplot(1, 3, 3)
                plt.plot(z0, np.real(Sz))
                plt.ylim(-S_lim, S_lim)
                plt.title("$S_z$")

            elif num_dims == 2:
                fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                         figsize=(1 * tx, 2 * ty))
                plt.subplot(3, 1, 1)
                plt.title("$S_x$")
                draw_field(np.real(Sx), self.x, self.z, cmap='seismic')
                format_drawing(self, plt, scale, draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)

                plt.subplot(3, 1, 2)
                plt.title("$S_y$")
                draw_field(np.real(Sy), self.x, self.z, cmap='seismic')
                format_drawing(self, plt, scale,  draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)

                plt.subplot(3, 1, 3)
                im3 = draw_field(np.real(Sz), self.x, self.z, cmap='seismic')
                format_drawing(self, plt, scale,  draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)
                plt.title("$S_z$")
                plt.xlabel('z ($\mu$m)')

                cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
                cbar = fig.colorbar(im3, cax=cb_ax, orientation='horizontal', shrink=0.5)

            # plt.suptitle("Instantaneous Poynting vector")

        return Sx, Sy, Sz


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def Poynting_vector_averaged(self, has_draw: bool = False, draw_borders: bool = True,  scale: str = ''):
        "Averaged Poynting Vector"

        Sx = np.real(self.Ey * self.Hz.conjugate() - self.Ez * self.Hy.conjugate()).squeeze()
        Sy = np.real(self.Ez * self.Hx.conjugate() - self.Ex * self.Hz.conjugate()).squeeze()
        Sz = np.real(self.Ex * self.Hy.conjugate() - self.Ey * self.Hx.conjugate()).squeeze()

        # if possible elliminate
        # Sz[0, :] = Sz[1, :]

        S_max = np.max((Sx, Sy, Sz))
        S_min = np.min((Sx, Sy, Sz))
        S_lim = np.max((abs(S_max), np.abs(S_min)))
        
        if has_draw:
            tx, ty = rcParams["figure.figsize"]

            dims = np.shape(Sx)
            num_dims = len(dims)
            if num_dims == 1:
                z0 = self.z
                plt.figure(figsize=(1 * tx, 3 * ty))
                plt.subplot(3, 1, 1)
                plt.plot(z0, Sx)
                plt.ylim(-S_lim, S_lim)
                plt.title("$S_x$")

                plt.subplot(3, 1, 2)
                plt.plot(z0, Sy)
                plt.title("$S_y$")
                plt.ylim(-S_lim, S_lim)

                plt.subplot(3, 1, 3)
                plt.plot(z0, Sz)
                plt.title("$S_z$")
                plt.ylim(-S_lim, S_lim)

                plt.suptitle("Average Pointing vector")

            elif num_dims == 2:
                z0 = self.z
                x0 = self.x

                fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,
                                         figsize=(1 * tx, 2 * ty))
                plt.subplot(3, 1, 1)
                plt.title("$S_x$")
                im1 = draw_field(Sx, x0, z0, cmap='seismic')
                format_drawing(self, plt, scale,  draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)
                # axes[0].set_axis_off()

                plt.subplot(3, 1, 2)
                plt.title("$S_y$")
                im2 = draw_field(Sy, x0, z0, cmap='seismic')
                format_drawing(self, plt, scale,  draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)
                # axes[1].set_axis_off()

                plt.subplot(3, 1, 3)
                im3 = draw_field(Sz, x0, z0, cmap='seismic')
                plt.title("$S_z$")
                format_drawing(self, plt, scale,  draw_borders, color='k.')
                plt.clim(-S_lim, S_lim)
                # axes[2].set_axis_off()

                cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
                cbar = fig.colorbar(im3, cax=cb_ax, orientation='horizontal', shrink=0.5)

        return Sx, Sy, Sz


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def Poynting_total(self, has_draw: bool = False, draw_borders: bool = True,  scale: str = ''):

        Sx, Sy, Sz = self.Poynting_vector_averaged(has_draw=False)

        S = np.sqrt(np.abs(Sx)**2 + np.abs(Sy)**2 + np.abs(Sz)**2)

        if has_draw:
            dims = np.shape(Sx)
            num_dims = len(dims)
            if num_dims == 1:
                plt.figure()
                plt.subplot(1, 1, 1)
                plt.plot(self.z, S)

                plt.suptitle("$S_{total}$")
            elif num_dims == 2:
                draw_field(S, self.x, self.z, cmap='hot')
                plt.colorbar(orientation='horizontal', shrink=0.5)
                format_drawing(self, plt, scale, draw_borders)
                plt.suptitle("$S_{total}$")
                plt.clim(vmin=0)

        return S


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def energy_density(self, has_draw: bool = False, draw_borders: bool = True,  scale: str = ''):

        epsilon = self.n ** 2
        permeability = 4 * np.pi * 1e-7

        U = epsilon * (np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2) + permeability * (np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)

        if has_draw:
            dims = np.shape(U)
            num_dims = len(dims)
            if num_dims == 1:
                plt.figure()
                plt.plot(self.z, np.real(U))

            elif num_dims == 2:

                draw_field(np.real(U), self.x, self.z, cmap='hot')
                plt.colorbar(orientation='horizontal', shrink=0.5)
                format_drawing(self, plt, scale, draw_borders)
                plt.clim(0)

            plt.title("energy_density")

        return U


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def irradiance(self, kind: str | tuple[float, float, float] = "modulus", has_draw: bool = False, draw_borders: bool = True,  scale: str = ''):
        """Irradiance of the field.
        
        
        The irradiance is defined as a scalar product between the Poynting vector and the normal vector to the surface.
        However here we determine the irradiance as the modulus of the Poynting vector ("modulus") or the z component of the Poynting vector ("Sz")


        Args:
            kind (str | tuple[float, float, float], optional): "Sz" or "modulus". Defaults to 'Sz'.
            has_draw (bool, optional): _description_. Defaults to False.
            axis (str, optional): _description_. Defaults to 'scaled'.

        Returns:
            _type_: _description_
        """

        epsilon = self.n ** 2
        permeability = 4 * np.pi * 1e-7

        Sx, Sy, Sz = self.Poynting_vector_averaged(has_draw=False)
        
        if kind == 'modulus':
            irradiance = np.sqrt(Sx**2 + Sy**2 + Sz**2)
            
        elif kind == 'Sz':
            irradiance = Sz
        elif isinstance(kind, (list, tuple, np.ndarray)):
            kind = np.array(kind)
            kind = kind/np.linalg.norm(kind)
            irradiance = kind[0] * Sx + kind[1] * Sy + kind[2] * Sz 

        if has_draw:
            dims = np.shape(irradiance)
            num_dims = len(dims)
            if num_dims == 1:
                plt.figure()
                plt.plot(self.z, irradiance)
                plt.ylim(0, irradiance.max())

            elif num_dims == 2:
                draw_field(irradiance, self.x, self.z, cmap='hot')
                plt.colorbar(orientation='horizontal', shrink=0.5)
                format_drawing(self, plt, scale, draw_borders)
                plt.clim(0, irradiance.max())

            plt.title("Irradiance: {}".format(kind))

        return irradiance


    def check_energy(self, kind = 'all', has_draw : bool = True):
        """
        check_energy. Integrates the Sz field and checks the energy conservation.

        We have used the z component of the Poynting vector.

        Args:
            kind (str): 'all', 'Sz', 'Stot', 'U'
            has_draw (bool, optional): If True, it draws the energy at each plane z. Defaults to True.

        Returns:
            np.array: normalized (to the first data) energy at each plane z.
        """
        
        # permeability = 4 * np.pi * 1e-7
        # Z0 = 376.82

        Sx, Sy, Sz = self.Poynting_vector_averaged(has_draw=False)
        U = self.energy_density(has_draw=False)
        S_tot = np.sqrt(Sx**2 + Sy**2 + Sz**2)



        energy_z1 = (Sz).mean(axis=1)/(Sz[0, :]).mean()
        energy_z2 = S_tot.mean(axis=1)/(S_tot[0, :]).mean()
        energy_z3 = (U/self.n).mean(axis=1)/(U[0, :]/self.n[0,:]).mean()

        if has_draw:
            plt.figure()
            if kind == 'all' or kind == 'Sz':
                plt.plot(self.z, energy_z1, 'r', label='Sz')
            if kind == 'all' or kind == 'Stot':
                plt.plot(self.z, energy_z2, 'g', label='St')
            if kind == 'all' or kind == 'U':
                plt.plot(self.z, energy_z3, 'b', label='u')

            plt.xlim(self.z[0], self.z[-1])
            plt.grid('on')
            plt.xlabel("$z\,(mm)$")
            plt.ylabel("$Sz(z)$")
            plt.ylim(bottom=0)
            plt.legend()


        return energy_z1, energy_z2, energy_z3


    @check_none('x','z','Ex','Ey',raise_exception=bool_raise_exception)
    def polarization_states(self, matrix: bool = False):
        """returns the Stokes parameters

        Args:
            Matrix (bool): if True returns Matrix, else Scalar_field_X

        Returns:
            S0,S1,S2,S3 images for Matrix=True
            S0,S1,S2,S3  for Matrix=False
        """

        I = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2
        Q = np.abs(self.Ex) ** 2 - np.abs(self.Ey) ** 2
        U = 2 * np.real(self.Ex * np.conjugate(self.Ey))
        V = 2 * np.imag(self.Ex * np.conjugate(self.Ey))

        if matrix is True:
            return I, Q, U, V
        else:
            CI = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            CQ = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            CU = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            CV = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)

            CI.u = I
            CQ.u = Q
            CU.u = U
            CV.u = V

            return CI, CQ, CU, CV


    @check_none('x','z',raise_exception=bool_raise_exception)
    def polarization_ellipse(self, pol_state=None, matrix: bool = False):
        """returns A, B, theta, h polarization parameter of elipses

        Args:
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
        L = Q + 1.0j * U + eps

        A = np.real(np.sqrt(0.5 * (Ip + np.abs(L) + eps)))
        B = np.real(np.sqrt(0.5 * (Ip - np.abs(L) + eps)))
        theta = 0.5 * np.angle(L)
        h = np.sign(V + eps)

        if matrix is True:
            return A, B, theta, h
        else:
            CA = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            CB = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            Ctheta = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
            Ch = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)

            CA.u = A
            CB.u = B
            Ctheta.u = theta
            Ch.u = h
            return (CA, CB, Ctheta, Ch)


    def normalize(self, kind='amplitude', new_field: bool = False):
        """Normalizes the field so that intensity.max()=1.

        Args:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
            kind (str): 'amplitude', or 'intensity'

        Returns
            u (numpy.array): normalized optical field
        """

        return normalize_field(self, kind, new_field)



    def draw(
        self,
        kind: Draw_Vector_XZ_Options = "intensity",
        logarithm: float = 0,
        normalize: bool = False,
        cut_value: float | None = None,
        draw_borders: bool = True,
        filename="",
        scale: str = '',
        percentage_intensity: float | None = None,
        draw=True,
        **kwargs
    ):
        """Draws electromagnetic field

        Args:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            filename (str): if not '' stores drawing in file,
            draw (bool): If True, it draws the field. Defaults to True.
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value


        """

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        if draw is True:
            if kind == "intensity":
                id_fig = self.__draw_intensity__(
                    logarithm, normalize, cut_value, draw_borders, scale, **kwargs
                )
            elif kind == "intensities":
                id_fig = self.__draw_intensities__(
                    logarithm, normalize, cut_value, draw_borders, scale, **kwargs
                )

            elif kind == "phases":
                id_fig = self.__draw_phases__(logarithm, normalize, cut_value, draw_borders, scale,  percentage_intensity, **kwargs)

            elif kind == "fields":
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value, draw_borders, scale,  percentage_intensity, **kwargs)

            elif kind == "EH":
                id_fig = self.__draw_EH__(logarithm, normalize, cut_value, draw_borders, scale, **kwargs)

            elif kind == "stokes":
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value, draw_borders, scale,  **kwargs)

            elif kind == "param_ellipses":
                id_fig = self.__draw_param_ellipse__(**kwargs)
            else:
                print("not good kind parameter in vector_fields_X.draw()")
                id_fig = None

            if filename != "":
                plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1)

            return id_fig


    def __draw_intensity__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        only_image=False,
        color_intensity=CONF_DRAWING["color_intensity"],
    ):
        """Draws the intensity

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get("intensity")

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.z, intensity)

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)


        plt.figure()
        h1 = plt.subplot(1, 1, 1)
        self.__draw1__(intensity, color_intensity, "", only_image=only_image)
        plt.colorbar(orientation='vertical', shrink=0.66)
        format_drawing(self, plt, scale, draw_borders)
            
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        # plt.tight_layout()
        return h1

    @check_none('x','z','Ex','Ey','Ez',raise_exception=bool_raise_exception)
    def __draw_intensities__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        only_image=False,
        color_intensity=CONF_DRAWING["color_intensity"],
        draw_z = True,
    ):
        """internal funcion: draws phase

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        tx, ty = rcParams["figure.figsize"]

        intensity1 = np.abs(self.Ex) ** 2
        intensity1 = normalize_draw(intensity1, logarithm, normalize, cut_value)

        intensity2 = np.abs(self.Ey) ** 2
        intensity2 = normalize_draw(intensity2, logarithm, normalize, cut_value)

        intensity3 = np.abs(self.Ez) ** 2
        intensity3 = normalize_draw(intensity3, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity1.max(), intensity2.max(), intensity3.max()))

        # percentage_z = 0.01

        # if intensity3.max() < percentage_z * intensity_max:
        if draw_z is False:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            self.__draw1__(intensity1, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(0, intensity_max)
            

            h2 = plt.subplot(1, 2, 2)
            self.__draw1__(intensity2, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2
        else:
            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            self.__draw1__(intensity1, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(0, intensity_max)
            format_drawing(self, plt, scale, draw_borders)

            h2 = plt.subplot(1, 3, 2)
            self.__draw1__(intensity2, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(0, intensity_max)

            h3 = plt.subplot(1, 3, 3)
            self.__draw1__(intensity3, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2, h3


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def __draw_phases__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        percentage_intensity: float | None = None,
        only_image: bool = False,
        color_intensity=CONF_DRAWING["color_phase"],
        draw_z: bool = True,
    ):
        """internal funcion: draws phase

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
            only_image: todo
            color_intensity: todo
            draw_z: todo
            percentage_intensity (None or number): If None it takes from CONF_DRAWING['percentage_intensity'], else uses this value
        """

        tx, ty = rcParams["figure.figsize"]

        intensity1 = np.abs(self.Ex) ** 2
        intensity1 = normalize_draw(intensity1, logarithm, normalize, cut_value)

        intensity2 = np.abs(self.Ey) ** 2
        intensity2 = normalize_draw(intensity2, logarithm, normalize, cut_value)

        intensity3 = np.abs(self.Ez) ** 2
        intensity3 = normalize_draw(intensity3, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity1.max(), intensity2.max(), intensity3.max()))

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        if draw_z is False:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 2, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(-180, 180)
            plt.tight_layout()

            return h1, h2
        else:
            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 3, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(-180, 180)

            h3 = plt.subplot(1, 3, 3)
            phase = np.angle(self.Ez)
            intensity = np.abs(self.Ez) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase/degrees, color_intensity, "", only_image=only_image)
            format_drawing(self, plt, scale, draw_borders)
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2, h3


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def __draw_fields__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        percentage_intensity: float | None = None,
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"],
        draw_z = True, ):
        """
        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        if percentage_intensity is None:
            percentage_intensity = percentage_intensity_config

        intensity_x = np.abs(self.Ex) ** 2
        intensity_x = normalize_draw(intensity_x, logarithm, normalize, cut_value)

        intensity_y = np.abs(self.Ey) ** 2
        intensity_y = normalize_draw(intensity_y, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity_x.max(), intensity_y.max()))
        tx, ty = rcParams["figure.figsize"]

        plt.figure(figsize=(2 * tx, 2 * ty))

        h1 = plt.subplot(2, 2, 1)

        self.__draw1__( intensity_x, color_intensity, "$I_x$")
        plt.clim(0, intensity_max)

        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(intensity_y, color_intensity,"$I_y$")
        format_drawing(self, plt, scale, draw_borders)
        plt.clim(0, intensity_max)

        h3 = plt.subplot(2, 2, 3)
        phase = np.angle(self.Ex)
        phase[intensity_x < percentage_intensity * (intensity_x.max())] = 0
        print(percentage_intensity)

        self.__draw1__(phase/degrees, color_phase, "$\phi_x$")
        format_drawing(self, plt, scale, draw_borders)
        plt.clim(-180, 180)

        h4 = plt.subplot(2, 2, 4)
        phase = np.angle(self.Ey)
        phase[intensity_y < percentage_intensity * (intensity_y.max())] = 0

        self.__draw1__(phase/degrees, color_phase, "$\phi_y$")
        format_drawing(self, plt, scale, draw_borders)
        plt.clim(-180, 180)
        h4 = plt.gca()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return h1, h2, h3, h4


    @check_none('x','z','Ex','Ey','Ez','Hx','Hy','Hz',raise_exception=bool_raise_exception)
    def __draw_EH__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        cmap=CONF_DRAWING["color_amplitude_sign"],
        edge=None,
        draw_z = True
    ):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Args:
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        E_x = self.Ex
        E_x = normalize_draw(E_x, logarithm, normalize, cut_value)

        E_y = self.Ey
        E_y = normalize_draw(E_y, logarithm, normalize, cut_value)

        E_z = self.Ez
        E_z = normalize_draw(E_z, logarithm, normalize, cut_value)

        H_x = self.Hx
        H_x = normalize_draw(H_x, logarithm, normalize, cut_value)

        H_y = self.Hy
        H_y = normalize_draw(H_y, logarithm, normalize, cut_value)

        H_z = self.Hz
        H_z = normalize_draw(H_z, logarithm, normalize, cut_value)

        tx, ty = rcParams["figure.figsize"]

        E_max = np.max((E_x.max(), E_y.max(), E_z.max()))
        H_max = np.max((H_x.max(), H_y.max(), H_z.max()))

        if draw_z is True:

            fig, axs = plt.subplots(
                nrows=3, ncols=2, sharex=True, sharey=True, figsize=(1.75 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], xlabel="", ylabel="$x  (\mu m)$", color=cmap, title=r'E$_x$')
            format_drawing(self, axs[0, 0], scale, draw_borders, color='k.')
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[1, 0], xlabel="", ylabel="$x  (\mu m)$", color=cmap, title=r'E$_y$')
            format_drawing(self, axs[1, 0], scale,  draw_borders, color='k.')
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_z, self.z, self.x, ax=axs[2, 0], xlabel="$z  (\mu m)$", ylabel="$x  (\mu m)$", color=cmap, title=r'E$_z$')
            format_drawing(self, axs[2, 0], scale,  draw_borders, color='k.')
            IDimage.set_clim(-E_max,E_max)
            # ax.colorbar()


            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[0, 1], xlabel="", ylabel="", color=cmap, title=r'H$_x$')
            format_drawing(self, axs[0, 1], scale,  draw_borders, color='k.')
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1], xlabel="", ylabel="", color=cmap, title=r'H$_y$')
            format_drawing(self, axs[1, 1], scale,  draw_borders, color='k.')
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_z, self.z, self.x, ax=axs[2, 1], xlabel="$z  (\mu m)$", ylabel="", color=cmap, title=r'H$_z$')
            # ax.colorbar()
            format_drawing(self, axs[2, 1], scale,  draw_borders, color='k.')
            IDimage.set_clim(-H_max,H_max)

            fig.subplots_adjust(right=1.25)
            cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
            fig.colorbar(IDimage, cax=cbar_ax)

            plt.tight_layout()
        else: 
            fig, axs = plt.subplots(
                nrows=2, ncols=2, sharex=True, sharey=True, figsize=(2 * tx, 2 * ty)
            )

            id_fig, ax, IDimage = draw2D_xz(
                E_x, self.z, self.x, ax=axs[0, 0], xlabel="", ylabel="$x  (\mu m)$", color=cmap, title=r'E$_x$')
            format_drawing(self, plt, scale,  draw_borders, color='k.')
            IDimage.set_clim(-E_max,E_max)
            id_fig, ax, IDimage = draw2D_xz(
                E_y, self.z, self.x, ax=axs[1, 0],  xlabel="$z  (\mu m)$", ylabel="$x  (\mu m)$", color=cmap, title=r'E$_y$')
            format_drawing(self, plt, scale,  draw_borders, color='k.')
            IDimage.set_clim(-E_max,E_max)


            id_fig, ax, IDimage = draw2D_xz(
                H_x, self.z, self.x, ax=axs[0, 1], xlabel="", ylabel="", color=cmap, title=r'H$_x$')
            format_drawing(self, plt, scale,  draw_borders, color='k.')
            IDimage.set_clim(-H_max,H_max)
            id_fig, ax, IDimage = draw2D_xz(
                H_y, self.z, self.x, ax=axs[1, 1],  xlabel="$z  (\mu m)$", ylabel="", color=cmap, title=r'H$_y$')
            format_drawing(self, plt, scale,  draw_borders, color='k.')
            IDimage.set_clim(-H_max,H_max)

            fig.subplots_adjust(right=1.25)
            cbar_ax = fig.add_axes([0.85, 0.25, 0.025, 0.5])
            fig.colorbar(IDimage, cax=cbar_ax)

            plt.tight_layout()
        return self

    def __draw_stokes__(
        self,
        logarithm,
        normalize,
        cut_value,
        draw_borders=False,
        scale = '',
        color_intensity=CONF_DRAWING["color_intensity"],
        color_stokes=CONF_DRAWING["color_stokes"],
    ):
        """__internal__: computes and draws CI, CQ, CU, CV parameters"""

        tx, ty = rcParams["figure.figsize"]

        S0, S1, S2, S3 = self.polarization_states(matrix=True)
        S0 = normalize_draw(S0, logarithm, normalize, cut_value)
        S1 = normalize_draw(S1, logarithm, normalize, cut_value)
        S2 = normalize_draw(S2, logarithm, normalize, cut_value)
        S3 = normalize_draw(S3, logarithm, normalize, cut_value)

        intensity_max = S0.max()

        plt.figure(figsize=(4 * tx, 1 * ty))
        h1 = plt.subplot(1,4,1)
        self.__draw1__(S0, color_intensity, "$S_0$")
        format_drawing(self, plt, scale,  draw_borders, color='w.')
        plt.clim(0, intensity_max)

        h2 = plt.subplot(1,4,2)
        self.__draw1__(S1, color_stokes, "$S_1$")
        format_drawing(self, plt, scale,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        h3 = plt.subplot(1,4,3)
        self.__draw1__(S2, color_stokes, "$S_2$")
        format_drawing(self, plt, scale,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        h4 = plt.subplot(1,4,4)
        self.__draw1__(S3, color_stokes, "$S_3$")
        format_drawing(self, plt, scale,  draw_borders, color='k.')
        plt.clim(-intensity_max, intensity_max)

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_param_ellipse__(
        self,
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"],
    ):
        """__internal__: computes and draws polariations ellipses.
        Args:
            color_intensity (_type_, optional): _description_. Defaults to CONF_DRAWING["color_intensity"].
            color_phase (_type_, optional): _description_. Defaults to CONF_DRAWING["color_phase"].

        Returns:
            _type_: _description_
        """

        A, B, theta, h = self.polarization_ellipse(pol_state=None, matrix=True)

        A = reduce_matrix_size(self.reduce_matrix, self.x, self.z, A)
        B = reduce_matrix_size(self.reduce_matrix, self.x, self.z, B)
        theta = reduce_matrix_size(self.reduce_matrix, self.x, self.z, theta)
        h = reduce_matrix_size(self.reduce_matrix, self.x, self.z, h)

        tx, ty = rcParams["figure.figsize"]

        plt.figure(figsize=(2 * tx, 2 * ty))

        max_intensity = max(A.max(), B.max())

        h1 = plt.subplot(2, 2, 1)
        self.__draw1__(A, color_intensity, "$A$")
        plt.clim(0, max_intensity)
        h2 = plt.subplot(2, 2, 2)
        self.__draw1__(B, color_intensity, "$B$")
        plt.clim(0, max_intensity)

        h3 = plt.subplot(2, 2, 3)
        self.__draw1__(theta/degrees, color_phase, "$\phi$")
        plt.clim(-180, 180)
        h4 = plt.subplot(2, 2, 4)
        self.__draw1__(h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_ellipses__(
        self,
        logarithm: float = 0.,
        normalize: bool = False,
        cut_value="",
        num_ellipses=(21, 21),
        amplification=0.75,
        color_line="w",
        line_width=1,
        draw_arrow=True,
        head_width=2,
        ax=False,
        color_intensity=CONF_DRAWING["color_intensity"],
    ):
        """

        Args:
            logarithm (float, optional): _description_. Defaults to 0..
            normalize (bool, optional): _description_. Defaults to False.
            cut_value (str, optional): _description_. Defaults to "".
            num_ellipses (tuple, optional): _description_. Defaults to (21, 21).
            amplification (float, optional): _description_. Defaults to 0.75.
            color_line (str, optional): _description_. Defaults to "w".
            line_width (int, optional): _description_. Defaults to 1.
            draw_arrow (bool, optional): _description_. Defaults to True.
            head_width (int, optional): _description_. Defaults to 2.
            ax (bool, optional): _description_. Defaults to False.
            color_intensity (_type_, optional): _description_. Defaults to CONF_DRAWING["color_intensity"].
        """

        percentage_intensity = CONF_DRAWING["percentage_intensity"]
        intensity_max = (np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2).max()

        Dx = self.x[-1] - self.x[0]
        Dy = self.z[-1] - self.z[0]
        size_x = Dx / (num_ellipses[0])
        size_y = Dy / (num_ellipses[1])
        x_centers = size_x/2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y/2 + size_y * np.array(range(0, num_ellipses[1]))

        num_x, num_y = len(self.x), len(self.z)
        ix_centers = num_x / (num_ellipses[0])
        iy_centers = num_y / (num_ellipses[1])

        ix_centers = (
            np.round(ix_centers/2 + ix_centers * np.array(range(0, num_ellipses[0])))
        ).astype("int")
        iy_centers = (
            np.round(iy_centers/2 + iy_centers * np.array(range(0, num_ellipses[1])))
        ).astype("int")

        Ix_centers, Iy_centers = np.meshgrid(
            ix_centers.astype("int"), iy_centers.astype("int")
        )

        verbose = False
        if verbose is True:
            print(num_x, num_y, ix_centers, iy_centers)
            print(Dx, Dy, size_x, size_y)
            print(x_centers, y_centers)
            print(Ix_centers, Iy_centers)

        E0x = self.Ex[Iy_centers, Ix_centers]
        E0y = self.Ey[Iy_centers, Ix_centers]

        angles = np.linspace(0, 360*degrees, 64)

        if ax is False:
            self.draw("intensity", logarithm=logarithm, color_intensity=color_intensity)
            ax = plt.gca()

        for i, xi in enumerate(ix_centers):
            for j, yj in enumerate(iy_centers):
                Ex = np.real(E0x[j, i] * np.exp(1j * angles))
                Ey = np.real(E0y[j, i] * np.exp(1j * angles))

                max_r = np.sqrt(np.abs(Ex) ** 2 + np.abs(Ey) ** 2).max()
                size_dim = min(size_x, size_y)

                if max_r > 0 and max_r**2 > percentage_intensity * intensity_max:
                    Ex = Ex / max_r * size_dim * amplification/2 + (+self.x[int(xi)])
                    Ey = Ey / max_r * size_dim * amplification/2 + self.z[int(yj)]

                    ax.plot(Ex, Ey, color_line, lw=line_width)
                    if draw_arrow:
                        ax.arrow(
                            Ex[0],
                            Ey[0],
                            Ex[0] - Ex[1],
                            Ey[0] - Ey[1],
                            width=0,
                            head_width=head_width,
                            fc=color_line,
                            ec=color_line,
                            length_includes_head=False,
                        )
                # else:
                #     print(max_r, intensity_max,
                #           percentage_intensity * intensity_max)

    def __draw1__(self, image, colormap, title: str = "", has_max=False, only_image=False):
        """_summary_

        Args:
            image (_type_): _description_
            colormap (_type_): _description_
            title (str, optional): _description_. Defaults to "".
            has_max (bool, optional): _description_. Defaults to False.
            only_image (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        extension = [self.z[0], self.z[-1], self.x[0], self.x[-1]]

        h = plt.imshow(
            image.transpose(),
            interpolation="bilinear",
            aspect="auto",
            origin="lower",
            extent=extension,
        )
        h.set_cmap(colormap)
        plt.axis(extension)

        if only_image is True:
            plt.axis("off")
            return h

        plt.title(title, fontsize=16)

        if has_max is True:
            text_up = "{}".format(image.max())
            plt.text(
                self.x.max(),
                self.z.max(),
                text_up,
                fontsize=14,
                bbox=dict(edgecolor="white", facecolor="white", alpha=0.75),
            )

            text_down = "{}".format(image.min())
            plt.text(
                self.x.max(),
                self.z.min(),
                text_down,
                fontsize=14,
                bbox=dict(edgecolor="white", facecolor="white", alpha=0.75),
            )

        plt.xlabel("$z  (\mu m)$")
        plt.ylabel("$x  (\mu m)$")
        if colormap is not None:
            plt.colorbar(orientation="horizontal", fraction=0.046, shrink=0.5)
            h.set_clim(0, image.max())

        return h


def polarization_ellipse(self, pol_state=None, matrix: bool = False):
    """returns A, B, theta, h polarization parameter of elipses

    Args:
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
    L = Q + 1.0j * U

    A = np.real(np.sqrt(0.5 * (Ip + np.abs(L))))
    B = np.real(np.sqrt(0.5 * (Ip - np.abs(L))))
    theta = 0.5 * np.angle(L)
    h = np.sign(V)

    if matrix is True:
        return A, B, theta, h
    else:
        CA = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
        CB = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
        Ctheta = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)
        Ch = Scalar_field_XZ(x=self.x, z=self.z, wavelength=self.wavelength)

        CA.u = A
        CB.u = B
        Ctheta.u = theta
        Ch.u = h
        return (CA, CB, Ctheta, Ch)


def FP_PWD_kernel_simple(Ex, Ey, n1, n2, k0, kx, wavelength, dz, has_H=True):
    """Step for Plane wave decomposition (PWD) algorithm.

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]
        has_H (bool, optional): If True computes magnetic field H. Defaults to True.

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Ex, Ey, Ez): Field H(z+dz) at at distance dz from the incident field.
        
    """

    # amplitude of waveplanes
    Exk = fftshift(fft(Ex))
    Eyk = fftshift(fft(Ey))


    kr = n1 * k0 # first layer
    ks = n2 * k0 # second layer
            
    ky = np.zeros_like(kx) # we are in XZ frame
    k_perp2 = kx**2 + ky**2

    kz_r = np.sqrt(kr**2 - k_perp2) # first layer
    kz_s = np.sqrt(ks**2 - k_perp2) # second layer

    P = np.exp(1j * kz_s * dz)
    Gamma = kz_r*kz_s + kz_s * k_perp2 / kz_r
    

    # Fresnel coefficients
    t_TM, t_TE, _, _ = fresnel_equations_kx(kx, wavelength, n1, n2, [1, 1, 0, 0], has_draw=False)
        
    T00 = P * (t_TM*kx**2*Gamma + t_TE*ky**2*kr*ks) / (k_perp2*kr*ks) 
    T01 = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks) 
    T10 = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks) 
    T11 = P * (t_TM*ky**2*Gamma + t_TE*kx**2*kr*ks) / (k_perp2*kr*ks) 
    
    # Simpler since ky = 0, but keep to translate to 3D 
    
    # T00 = P * (t_TM*kx**2*Gamma) / (k_perp2*kr*ks) 
    # T01 = np.zeros_like(kx) 
    # T10 = np.zeros_like(kx)  
    # T11 = P * (t_TE*kx**2*kr*ks) / (k_perp2*kr*ks) 
    
    nan_indices = np.where(np.isnan(T00)) # TODO: fix better
    
    option = 1
    
    
    if option == 1:

        T00[nan_indices]=T00[nan_indices[0]-1]
        T01[nan_indices]=T01[nan_indices[0]-1]
        T10[nan_indices]=T10[nan_indices[0]-1]
        T11[nan_indices]=T11[nan_indices[0]-1] 
        
    elif option == 2:
    
        T00_b = P * (t_TM*kx**2*Gamma + t_TE*ky**2*kr*ks) / (k_perp2*kr*ks+1e-10) 
        T01_b = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks+1e-10) 
        T10_b = P * (t_TM*kx*ky*Gamma - t_TE*kx*ky*kr*ks) / (k_perp2*kr*ks+1e-10) 
        T11_b = P * (t_TM*ky**2*Gamma + t_TE*kx**2*kr*ks) / (k_perp2*kr*ks+1e-10) 
    
        T00[nan_indices]=T00_b[nan_indices[0]]
        T01[nan_indices]=T01_b[nan_indices[0]]
        T10[nan_indices]=T10_b[nan_indices[0]]
        T11[nan_indices]=T11_b[nan_indices[0]] 
    

   
    ex0 = T00 * Exk + T01 * Eyk
    ey0 = T10 * Exk + T11 * Eyk 
    ez0 = - (kx*ex0+ky*ey0) / (kz_s)
    
    # ex0 = T00 * Exk 
    # ey0 = T11 * Eyk 
    # ez0 = - (kx*ex0+ky*ey0) / (kz_r)
    

    if has_H:
        
        # thesis Fertig 2011 (3.40) pág 66 I do not feel confident yet
        TM00 = -kx*ky*Gamma 
        TM01 = -(ky*ky*Gamma + kz_s**2)
        TM10 = +(kx*kx*Gamma + kz_s**2)
        TM11 = +kx*ky*Gamma
        TM20 = -ky*kz_s
        TM21 = +kx*kz_s
        
        Z0 = 376.82  # ohms (impedance of free space)
        H_factor = n2 / (ks * kz_s * Z0)
        
        hx0 = (TM00*ex0+TM01*ey0) * H_factor
        hy0 = (TM10*ex0+TM11*ey0) * H_factor
        hz0 = (TM20*ex0+TM21*ey0) * H_factor
        
        # hx0 = -(kz_s**2) * ey0 * H_factor
        # hy0 =  (kz_s**2) * ex0 * H_factor  # cuidado, ver (3.16) pág 60 y (3.40) pág 66 de tesis VWPM
        # hz0 =  (kx*kz_s) * ey0 * H_factor
        
    else:
        Hx_final, Hy_final, Hz_final = 0.0, 0.0, 0.0

    Ex_final = ifft(ifftshift(ex0))
    Ey_final = ifft(ifftshift(ey0))
    Ez_final = ifft(ifftshift(ez0))


    Hx_final = ifft(ifftshift(hx0))
    Hy_final = ifft(ifftshift(hy0))
    Hz_final = ifft(ifftshift(hz0))

    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)



def FP_WPM_schmidt_kernel(Ex, Ey, n1, n2, k0, kx, wavelength, dz, has_H=True):
    """
    Kernel for fast propagation of WPM method

    Args:
        Ex (np.array): field Ex
        Ey (np.array): field Ey
        n1 (np.array): refractive index at the first layer
        n2 (np.array): refractive index at the second layer
        k0 (float): wavenumber
        kx (np.array): transversal wavenumber
        wavelength (float): wavelength
        dz (float): increment in distances: z[1]-z[0]
        has_H (bool, optional): If True computes magnetic field H. Defaults to True.

    Returns:
        E  list(Ex, Ey, Ez): Field E(z+dz) at at distance dz from the incident field.
        H  list(Hx, Hy, Hz): Field H(z+dz) at at distance dz from the incident field.

    References:

        1. M. W. Fertig and K.-H. Brenner, “Vector wave propagation method,” J. Opt. Soc. Am. A, vol. 27, no. 4, p. 709, 2010.

        2. S. Schmidt et al., “Wave-optical modeling beyond the thin-element-approximation,” Opt. Express, vol. 24, no. 26, p. 30188, 2016.
    """
    Nr = np.unique(n1)
    Ns = np.unique(n2)

    Ex_final = np.zeros_like(Ex, dtype=complex)
    Ey_final = np.zeros_like(Ey, dtype=complex)
    Ez_final = np.zeros_like(Ey, dtype=complex)

    if has_H:
        Hx_final = np.zeros_like(Ex, dtype=complex)
        Hy_final = np.zeros_like(Ey, dtype=complex)
        Hz_final = np.zeros_like(Ey, dtype=complex)
    else:
        Hx_final = 0
        Hy_final = 0
        Hz_final = 0

    for r, n_r in enumerate(Nr):
        for s, n_s in enumerate(Ns):
            Imz = np.array(np.logical_and(n1 == n_r, n2 == n_s))
            E, H = FP_PWD_kernel_simple(Ex, Ey, n_r, n_s, k0, kx, wavelength, dz, has_H)

            Ex_final = Ex_final + Imz * E[0]
            Ey_final = Ey_final + Imz * E[1]
            Ez_final = Ez_final + Imz * E[2]
            Hx_final = Hx_final + Imz * H[0]
            Hy_final = Hy_final + Imz * H[1]
            Hz_final = Hz_final + Imz * H[2]
    return (Ex_final, Ey_final, Ez_final), (Hx_final, Hy_final, Hz_final)


def draw_field(u, x_f, z_f, axis=None, interpolation='bilinear', cmap=None):
    extent = [z_f[0], z_f[-1], x_f[0], x_f[-1]]
    im = plt.imshow(u.transpose(),
                    origin='lower',
                    aspect='auto',
                    interpolation=interpolation,
                    extent=extent,
                    cmap=cmap)
    if axis is not None:
        plt.axis(axis)
    return im


def draw2D_xz(
        image,
        x,
        y,
        ax=None,
        xlabel="$x  (\mu m)$",
        ylabel="$y  (\mu m)$",
        title="",
        color="YlGnBu",  # YlGnBu  seismic
        interpolation='bilinear',  # 'bilinear', 'nearest'
        scale='scaled',
        reduce_matrix='standard',
        range_scale='um',
        verbose=False):
    """makes a drawing of XY

    Args:
        image (numpy.array): image to draw
        x (numpy.array): positions x
        y (numpy.array): positions y
        ax (): axis
        xlabel (str): label for x
        ylabel (str): label for y
        title (str): title
        color (str): color
        interpolation (str): 'bilinear', 'nearest'
        scale (str): kind of axis (None, 'equal', 'scaled', etc.)
        range_scale (str): 'um' o 'mm'
        verbose (bool): if True prints information

    Returns:
        id_fig: handle of figure
        IDax: handle of axis
        IDimage: handle of image
    """
    if reduce_matrix in (None, '', []):
        pass
    elif reduce_matrix == 'standard':
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_y / 500)

        if reduction_x == 0:
            reduction_x = 1
        if reduction_y == 0:
            reduction_y = 1

        image = image[::reduction_x, ::reduction_y]
    else:
        image = image[::reduce_matrix[0], ::reduce_matrix[1]]

    if verbose is True:
        print(("image size {}".format(image.shape)))

    if ax is None:
        id_fig = plt.figure()
        ax = id_fig.add_subplot(111)
    else:
        id_fig = None

    if range_scale == 'um':
        extension = (x[0], x[-1], y[0], y[-1])
    else:
        extension = (x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm)
        xlabel = "x (mm)"
        ylabel = "y (mm)"

    IDimage = ax.imshow(image.transpose(),
                        interpolation=interpolation,
                        aspect='auto',
                        origin='lower',
                        extent=extension,
                        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if scale != '':
        ax.axis(scale)
    # plt.axis(extension)
    # if scale not in ('', None, []):
    #     ax.set_axis(scale)
    IDimage.set_cmap(color)

    return id_fig, ax, IDimage


def format_drawing(vector_field_XZ, plt, scale, draw_borders, color='w.', ms=0.05) -> None:
    
        if scale != '':
            plt.axis(scale)

        if draw_borders is True:
            min_incr: float = 0.0005,
            vector_field_XZ.surface_detection(1, min_incr, reduce_matrix=False)
            border0 = vector_field_XZ.borders[0]
            border1 = vector_field_XZ.borders[1]

            plt.plot(border1, border0, color, ms=ms)
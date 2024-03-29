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
import time
from scipy.interpolate import RectBivariateSpline

from . import degrees, eps, mm, np, plt
from .config import CONF_DRAWING
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_XZ import Scalar_mask_XZ
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw, reduce_matrix_size
from .utils_math import get_k, nearest
from .utils_optics import normalize_field, fresnel_equations_kx


from numpy.lib.scimath import sqrt as csqrt
from scipy.fftpack import fft, fftshift, ifft, ifftshift

percentage_intensity = CONF_DRAWING["percentage_intensity"]

# From Scalar_mask_XZ to have refraction index.
# TODO: anistropic masks.
class Vector_field_XZ(Scalar_mask_XZ):
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

    def __init__(self, x, z, wavelength, n_background=1, info=""):
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
        self.borders = None  # borders at refraction index

        
        self.Ex0 = np.zeros_like(self.x)
        self.Ey0 = np.zeros_like(self.x)
        
        

        self.reduce_matrix = "standard"  # 'None, 'standard', (5,5)
        self.type = "Vector_field_XZ"
        self.info = info
        self.date = get_date()
        self.CONF_DRAWING = CONF_DRAWING

    def __str__(self):
        """Represents data from class."""

        intensity = self.intensity()
        Imin = intensity.min()
        Imax = intensity.max()

        print(
            "{}\n - x:  {},     Ex:  {}".format(self.type, self.x.shape, self.Ex.shape)
        )
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

    def __add__(self, other, kind="standard"):
        """adds two Vector_field_X. For example two light sources or two masks

        Parameters:
            other (Vector_field_X): 2nd field to add
            kind (str): instruction how to add the fields:

        Returns:
            Vector_field_X: `E3 = E1 + E2`
        """

        EM = Vector_field_XZ(self.x, self.z, self.wavelength)

        if kind == "standard":
            EM.Ex = self.Ex + other.Ex
            EM.Ey = self.Ey + other.Ey
            EM.Ez = self.Ez + other.Ez

        return EM

    def save_data(self, filename, add_name="", description="", verbose=False):
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
            final_filename = save_data_common(
                self, filename, add_name, description, verbose
            )
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
                raise Exception("no dictionary in load_data")

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
    
    
    def incident_field(self, E0, z0=None):
        """Incident field for the experiment. It takes a Scalar_source_X field

        Parameters:
            E0 (Vector_source_X): field produced by Scalar_source_X (or a X field)
            z0 (float): position of the incident field. if None, '', [], is at the beginning
        """

        if z0 in (None, '', []):
            self.Ex0 = E0.Ex
            self.Ey0 = E0.Ey
            
            self.Ex[:, 0] = self.Ex[:, 0] + E0.Ex
            self.Ey[:, 0] = self.Ey[:, 0] + E0.Ey
            
        else:
            iz, _, _ = nearest(self.z, z0)
            self.Ex[:, iz] = self.Ex[:, iz] + E0.Ex
            self.Ey[:, iz] = self.Ey[:, iz] + E0.Ey

    def final_field(self):
        """Returns the final field as a Vector_field_X."""

        EH_final = Vector_field_X(x=self.x,
                                 wavelength=self.wavelength,
                                 n_background=self.n_background,
                                 info="from final_field at z0= {} um".format(
                                     self.z[-1]))
        EH_final.Ex = self.Ex[:, -1]
        EH_final.Ey = self.Ey[:, -1]
        EH_final.Ez = self.Ez[:, -1]
        EH_final.Hx = self.Hx[:, -1]
        EH_final.Hy = self.Hy[:, -1]
        EH_final.Hz = self.Hz[:, -1]   
        return EH_final

    def get(self, kind="fields", is_matrix=True):
        """Takes the vector field and divide in Scalar_field_X.

        Parameters:
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

    def apply_mask(self, u):
        """Multiply field by binary scalar mask: self.Ex = self.Ex * u.u

        Parameters:
           u (Scalar_mask_X): mask
        """
        self.Ex = self.Ex * u.u
        self.Ey = self.Ey * u.u
        self.Ez = self.Ez * u.u

    def FP_WPM(
        self, has_edges=True, pow_edge=80, matrix=False, has_H=True, verbose=False
    ):
        """
        WPM Method. 'schmidt methodTrue is very fast, only needs discrete number of refraction indexes'


        Parameters:
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

        # self.Ex[:, 0] = self.Ex0
        # self.Ey[:, 0] = self.Ey0

        if has_H:
            self.Hx = np.zeros_like(self.Ex)
            self.Hy = np.zeros_like(self.Ex)
            self.Hz = np.zeros_like(self.Ex)

        kx = get_k(x, flavour="+")

        if has_edges is False:
            has_filter = np.zeros_like(self.z)

        if isinstance(has_edges, int):
            has_filter = np.ones_like(self.z)
        else:
            has_filter = has_edges

        width_edge = (self.x[-1] - self.x[0]) / 2
        x_center = (self.x[-1] + self.x[0]) / 2

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
            )


            self.Ex[j, :] = self.Ex[j, :] + E_step[0] * filter_edge
            self.Ey[j, :] = self.Ey[j, :] + E_step[1] * filter_edge
            self.Ez[j, :] = E_step[2] * filter_edge

            if has_H:
                self.Hx[j, :] = H_step[0] * filter_edge
                self.Hy[j, :] = H_step[1] * filter_edge
                self.Hz[j, :] = H_step[2] * filter_edge

        t2 = time.time_ns()
        if verbose is True:
            print(
                "Time = {:2.2f} s, time/loop = {:2.4} ms".format(
                    (t2 - t1) / 1e9, (t2 - t1) / len(self.z) / 1e6
                )
            )

        if matrix is True:
            return (self.Ex, self.Ey, self.Ez), (self.Hx, self.Hy, self.Hz)


    def intensity(self):
        """ "Returns intensity."""
        intensity = np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2

        return intensity




    def Poynting_vector(self, has_draw=True, axis='equal'):
        "Instantaneous Poynting Vector"

        Sx = self.Ey * self.Hz - self.Ez * self.Hy
        Sy = self.Ez * self.Hx - self.Ex * self.Hz
        Sz = self.Ex * self.Hy - self.Ey * self.Hx

        S_max = np.real(np.max((Sx, Sy, Sz)))
        S_min = np.real(np.min((Sx, Sy, Sz)))
        S_lim = np.max((abs(S_max), np.abs(S_min)))

        if has_draw:
            dims = np.shape(self.Ex)
            num_dims = len(dims)
            if num_dims == 1:
                z0 = self.z

                plt.figure()
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
                fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
                plt.subplot(3, 1, 1)
                plt.title("$S_x$")
                draw_field(np.real(Sx), self.x, self.z, axis, cmap='seismic')
                plt.clim(-S_lim, S_lim)

                plt.subplot(3, 1, 2)
                plt.title("$S_y$")
                draw_field(np.real(Sy), self.x, self.z, axis, cmap='seismic')
                plt.clim(-S_lim, S_lim)

                plt.subplot(3, 1, 3)
                im3 = draw_field(np.real(Sz), self.x, self.z, axis, cmap='seismic')
                plt.clim(-S_lim, S_lim)
                plt.title("$S_z$")
                plt.xlabel('z ($\mu$m)')

                cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
                cbar = fig.colorbar(im3, cax=cb_ax, orientation='horizontal')

            plt.suptitle("Instantaneous Poynting vector")

        return Sx, Sy, Sz


    def Poynting_vector_averaged(self, has_draw=False, axis='scaled'):
        "Averaged Poynting Vector"


        Sx = np.real(self.Ey * self.Hz.conjugate() - self.Ez * self.Hy.conjugate()).squeeze()
        Sy = np.real(self.Ez * self.Hx.conjugate() - self.Ex * self.Hz.conjugate()).squeeze()
        Sz = np.real(self.Ex * self.Hy.conjugate() - self.Ey * self.Hx.conjugate()).squeeze()
        
        # if possible elliminate
        Sz[0,:]=Sz[1,:]

        S_max = np.max((Sx, Sy, Sz))
        S_min = np.min((Sx, Sy, Sz))
        S_lim = np.max((abs(S_max), np.abs(S_min)))

        if has_draw:
            dims = np.shape(Sx)
            num_dims = len(dims)
            if num_dims == 1:
                z0 = self.z
                plt.figure()
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

                fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
                plt.subplot(3, 1, 1)
                plt.title("$S_x$")
                im1 = draw_field(Sx, x0, z0, axis, cmap='seismic')
                plt.clim(-S_lim, S_lim)
                #axes[0].set_axis_off()

                plt.subplot(3, 1, 2)
                plt.title("$S_y$")
                im2 = draw_field(Sy, x0, z0, axis, cmap='seismic')
                plt.clim(-S_lim, S_lim)
                #axes[1].set_axis_off()

                plt.subplot(3, 1, 3)
                im3 = draw_field(Sz, x0, z0, axis, cmap='seismic')
                plt.title("$S_z$")
                plt.clim(-S_lim, S_lim)
                #axes[2].set_axis_off()

                # = fig.colorbar(im3, ax=axes.ravel().tolist(), shrink=0.95)
                cb_ax = fig.add_axes([0.1, 0, 0.8, 0.05])
                cbar = fig.colorbar(im3, cax=cb_ax, orientation='horizontal')

        return Sx, Sy, Sz


    def Poynting_total(self, has_draw=False, axis='scaled'):

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
                draw_field(S, self.x, self.z, axis, cmap='hot')
                plt.colorbar(orientation='horizontal')
                plt.suptitle("$S_{total}$")
                plt.clim(vmin=0)

        return S


    def energy_density(self, has_draw=False, axis='scaled'):

        epsilon = self.n **2
        permeability = 4 * np.pi * 1e-7
        

        U = epsilon * (np.abs(self.Ex)**2 + np.abs(self.Ey)**2 + np.abs(self.Ez)**2) + permeability * (np.abs(self.Hx)**2 + np.abs(self.Hy)**2 + np.abs(self.Hz)**2)

        if has_draw:
            dims = np.shape(U)
            num_dims = len(dims)
            if num_dims == 1:
                plt.figure()
                plt.plot(self.z, np.real(U))

            elif num_dims == 2:

                draw_field(np.real(U), self.x, self.z, axis, cmap='hot')
                plt.colorbar(orientation='horizontal')

            plt.title("energy_density")
        
    
            
        return U



    def irradiance(self, has_draw=False, axis='scaled'):

        epsilon = self.n **2
        permeability = 4 * np.pi * 1e-7

        Sx, Sy, Sz = self.Poynting_vector_averaged(has_draw=False)

        if has_draw:
            dims = np.shape(Sz)
            num_dims = len(dims)
            if num_dims == 1:
                plt.figure()
                plt.plot(self.z, np.Sz)

            elif num_dims == 2:

                draw_field(Sz, self.x, self.z, axis, cmap='hot')
                plt.colorbar(orientation='horizontal')
                plt.clim(0,Sz.max())

            plt.title("Irradiance")
        
    
            
        return Sz


    def check_energy(self, I0=None):
        permeability = 4 * np.pi * 1e-7
        Z0 = 376.82
        
        Sx, Sy, Sz = self.Poynting_vector_averaged(has_draw=False)
        U = self.energy_density(has_draw=False)

        check_Sz = Sz.mean(axis=1)/Sz[0,:].mean()
        check_U = U.mean(axis=1)/U[0,:].mean()

        plt.figure()
        plt.plot(self.z, check_Sz, 'r', label='Sz')
        #plt.plot(self.z, check_U, 'b', label='U')
        plt.legend()
        
        plt.xlim(self.z[0], self.z[-1])
        plt.grid('on')
        
        if I0 is not None:
            plt.ylim(ymin=I0)
        
        return check_Sz #, check_U
# 

    def polarization_states(self, matrix=False):
        """returns the Stokes parameters

        Parameters:
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

    def normalize(self, new_field=False):
        """Normalizes the field so that intensity.max()=1.

        Parameters:
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
        Returns
            u (numpy.array): normalized optical field
        """
        return normalize_field(self, new_field)

    def draw(
        self,
        kind="intensity",
        logarithm=0,
        normalize=False,
        cut_value=None,
        filename="",
        draw=True,
        **kwargs
    ):
        """Draws electromagnetic field

        Parameters:
            kind (str):  'intensity', 'intensities', intensities_rz, 'phases', fields', 'stokes'
            logarithm (float): If >0, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            filename (str): if not '' stores drawing in file,

        """

        if draw is True:
            if kind == "intensity":
                id_fig = self.__draw_intensity__(
                    logarithm, normalize, cut_value, **kwargs
                )
            elif kind == "intensities":
                id_fig = self.__draw_intensities__(
                    logarithm, normalize, cut_value, **kwargs
                )

            elif kind == "phases":
                id_fig = self.__draw_phases__(logarithm, normalize, cut_value, **kwargs)

            elif kind == "fields":
                id_fig = self.__draw_fields__(logarithm, normalize, cut_value, **kwargs)

            elif kind == "stokes":
                id_fig = self.__draw_stokes__(logarithm, normalize, cut_value, **kwargs)

            elif kind == "param_ellipses":
                id_fig = self.__draw_param_ellipse__(
                    logarithm, normalize, cut_value, **kwargs
                )

            else:
                print("not good kind parameter in vector_fields_X.draw()")
                id_fig = None

            if not filename == "":
                plt.savefig(filename, dpi=100, bbox_inches="tight", pad_inches=0.1)

            return id_fig

    def __draw_intensity__(
        self,
        logarithm,
        normalize,
        cut_value,
        only_image=False,
        color_intensity=CONF_DRAWING["color_intensity"],
    ):
        """Draws the intensity

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
        """

        intensity = self.get("intensity")

        intensity = reduce_matrix_size(self.reduce_matrix, self.x, self.z, intensity)

        intensity = normalize_draw(intensity, logarithm, normalize, cut_value)

        plt.figure()
        h1 = plt.subplot(1, 1, 1)
        self.__draw1__(intensity, color_intensity, "", only_image=only_image)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return h1

    def __draw_intensities__(
        self,
        logarithm,
        normalize,
        cut_value,
        only_image=False,
        color_intensity=CONF_DRAWING["color_intensity"],
    ):
        """internal funcion: draws phase

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
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

        percentage_z = 0.01

        if intensity3.max() < percentage_z * intensity_max:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            self.__draw1__(intensity1, color_intensity, "", only_image=only_image)
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 2, 2)
            self.__draw1__(intensity2, color_intensity, "", only_image=only_image)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2
        else:
            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            self.__draw1__(intensity1, color_intensity, "", only_image=only_image)
            plt.clim(0, intensity_max)

            h2 = plt.subplot(1, 3, 2)
            self.__draw1__(intensity2, color_intensity, "", only_image=only_image)
            plt.clim(0, intensity_max)

            h3 = plt.subplot(1, 3, 3)
            self.__draw1__(intensity3, color_intensity, "", only_image=only_image)
            plt.clim(0, intensity_max)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_phases__(
        self,
        logarithm,
        normalize,
        cut_value,
        only_image=False,
        color_intensity=CONF_DRAWING["color_phase"],
    ):
        """internal funcion: draws phase

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
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

        percentage_z = 0.01

        if intensity3.max() < percentage_z * intensity_max:
            plt.figure(figsize=(2 * tx, ty))

            h1 = plt.subplot(1, 2, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees, color_intensity, "", only_image=only_image)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 2, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees, color_intensity, "", only_image=only_image)
            plt.clim(-180, 180)
            plt.tight_layout()

            return h1, h2
        else:
            plt.figure(figsize=(3 * tx, ty))

            h1 = plt.subplot(1, 3, 1)
            phase = np.angle(self.Ex)
            intensity = np.abs(self.Ex) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees, color_intensity, "", only_image=only_image)
            plt.clim(-180, 180)

            h2 = plt.subplot(1, 3, 2)
            phase = np.angle(self.Ey)
            intensity = np.abs(self.Ey) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees, color_intensity, "", only_image=only_image)
            plt.clim(-180, 180)

            h3 = plt.subplot(1, 3, 3)
            phase = np.angle(self.Ez)
            intensity = np.abs(self.Ez) ** 2
            phase[intensity < percentage_intensity * (intensity.max())] = 0

            self.__draw1__(phase / degrees, color_intensity, "", only_image=only_image)
            plt.clim(-180, 180)

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
            plt.tight_layout()

            return h1, h2, h3

    def __draw_fields__(
        self,
        logarithm,
        normalize,
        cut_value,
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"],
    ):
        """__internal__: draws amplitude and phase in 2x2 drawing

        Parameters:
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            title (str): title of figure
            cut_value (float): If not None, cuts the maximum intensity to this value

        """

        intensity_x = np.abs(self.Ex) ** 2
        intensity_x = normalize_draw(intensity_x, logarithm, normalize, cut_value)

        intensity_y = np.abs(self.Ey) ** 2
        intensity_y = normalize_draw(intensity_y, logarithm, normalize, cut_value)

        intensity_max = np.max((intensity_x.max(), intensity_y.max()))
        tx, ty = rcParams["figure.figsize"]

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
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return h1, h2, h3, h4

    def __draw_stokes__(
        self,
        logarithm,
        normalize,
        cut_value,
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

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0)
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_param_ellipse__(
        self,
        color_intensity=CONF_DRAWING["color_intensity"],
        color_phase=CONF_DRAWING["color_phase"],
    ):
        """__internal__: computes and draws polariations ellipses"""
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
        self.__draw1__(theta / degrees, color_phase, "$\phi$")
        plt.clim(-180, 180)
        h4 = plt.subplot(2, 2, 4)
        self.__draw1__(h, "gist_heat", "$h$")
        plt.tight_layout()
        return (h1, h2, h3, h4)

    def __draw_ellipses__(
        self,
        logarithm=False,
        normalize=False,
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
        """__internal__: draw ellipses

        Parameters:
            num_ellipses (int): number of ellipses for parameters_ellipse
        """

        percentage_intensity = CONF_DRAWING["percentage_intensity"]
        intensity_max = (np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2).max()

        Dx = self.x[-1] - self.x[0]
        Dy = self.z[-1] - self.z[0]
        size_x = Dx / (num_ellipses[0])
        size_y = Dy / (num_ellipses[1])
        x_centers = size_x / 2 + size_x * np.array(range(0, num_ellipses[0]))
        y_centers = size_y / 2 + size_y * np.array(range(0, num_ellipses[1]))

        num_x, num_y = len(self.x), len(self.z)
        ix_centers = num_x / (num_ellipses[0])
        iy_centers = num_y / (num_ellipses[1])

        ix_centers = (
            np.round(ix_centers / 2 + ix_centers * np.array(range(0, num_ellipses[0])))
        ).astype("int")
        iy_centers = (
            np.round(iy_centers / 2 + iy_centers * np.array(range(0, num_ellipses[1])))
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

        angles = np.linspace(0, 360 * degrees, 64)

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
                    Ex = Ex / max_r * size_dim * amplification / 2 + (+self.x[int(xi)])
                    Ey = Ey / max_r * size_dim * amplification / 2 + self.z[int(yj)]

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

    def __draw1__(self, image, colormap, title="", has_max=False, only_image=False):
        """Draws image

        Parameters:
            image (numpy.array): array with drawing
            colormap (str): colormap
            title (str): title of drawing
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
            plt.colorbar(orientation="horizontal", fraction=0.046)
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
        CA = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        CB = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        Ctheta = Scalar_field_X(x=self.x, wavelength=self.wavelength)
        Ch = Scalar_field_X(x=self.x, wavelength=self.wavelength)
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
        n1 (np.array): refraction index at the first layer
        n2 (np.array): refraction index at the second layer
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


    kr = n1 * k0
    ks = n2 * k0

    kz_r = csqrt(kr**2 - kx**2)
    kz_s = csqrt(ks**2 - kx**2)

    # TODO: Solve Gamma.
    # Gamma = kz_r.conjugate()*kz_s + kz_s*kx**2 / kz_r
    # Gamma = kz_r*kz_s + kz_s*kx**2 / kz_r

    # Fresnel coefficients
    t_TM, t_TE, _, _ = fresnel_equations_kx(
        kx, wavelength, n1, n2, [1, 1, 0, 0], has_draw=False
    )

    # y_symmetry. No interaction between TE and TM
    T00 = t_TM
    # T01 = 0
    # T10 = 0
    T11 = t_TE
    # aquí ya está metido el divisor

    P = np.exp(1j * dz * kz_s)

    T2_00 = P * T00
    # T2_01 = P * T01
    # T2_10 = P * T10
    T2_11 = P * T11

    ex0 = T2_00 * Exk
    ey0 = T2_11 * Eyk
    ez0 = -(kx / kz_s) * T2_00 * Exk

    if has_H:
        Z0 = 376.82  # ohms (proportional m**2)
        H_factor = n2 / (ks * kz_s * Z0)
        hx0 = -(kz_s**2) * ey0 * H_factor
        hy0 = (
            (kz_s**2) * ex0 * H_factor
        )  # cuidado, ver (3.16) pág 60 y (3.40) pág 66 de tesis VWPM
        hz0 = (kx * kz_s) * ey0 * H_factor
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
        n1 (np.array): refraction index at the first layer
        n2 (np.array): refraction index at the second layer
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



def draw_field(u, x_f, z_f, axis, interpolation='bilinear', cmap=None):
    extent = [z_f[0], z_f[-1], x_f[0], x_f[-1]]
    im = plt.imshow(u.transpose(),
                    origin='lower',
                    interpolation=interpolation,
                    extent=extent,
                    cmap=cmap)
    plt.axis(axis)
    return im

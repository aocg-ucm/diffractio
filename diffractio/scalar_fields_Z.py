# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_field_Z class

The main atributes are:
    * self.z (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
    * self.wavelength (float): wavelength of the incident field.
    * self.u (numpy.array): equal size than  x. complex field

There are also some secondary atributes:
    * self.quality (float): quality of RS algorithm
    * self.info (str): description of data
    * self.type (str): Class of the field
    * self.date (str): date


*Class for unidimensional scalar fields*

*Definition of a scalar field*
    * instantiation, duplicate,  clear_field, print
    * save and load data



*Drawing functions*
    * draw

*Parameters:*
    * intensity, average intensity
    * get_edges_transitions (mainly for pylithography)

"""

import copy
import multiprocessing

from numpy import (angle, exp, linspace, pi, shape, zeros)
from scipy.interpolate import interp1d

from . import degrees, mm, np, plt
from .utils_common import get_date, load_data_common, save_data_common
from .utils_drawing import normalize_draw
from .utils_math import nearest

from .utils_optics import field_parameters, normalize, FWHM1D


num_max_processors = multiprocessing.cpu_count()


class Scalar_field_Z(object):
    """Class for unidimensional scalar fields in z axis.

    Parameters:
        z (numpy.array): linear array with equidistant positions.
        wavelength (float): wavelength of the incident field
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.z (numpy.array): Linear array with equidistant positions.
            The number of data is preferibly :math:`2^n`.
        self.wavelength (float): Wavelength of the incident field.
        self.u (numpy.array): Complex field. The size is equal to self.z.
        self.quality (float): Quality of RS algorithm.
        self.info (str): Description of data.
        self.type (str): Class of the field.
        self.date (str): Date when performed.
    """

    def __init__(self, z=None, wavelength=None, n_background=1, info=""):
        self.z = z
        self.wavelength = wavelength
        self.n_background = n_background
        if z is not None:
            self.u = zeros(shape(self.z), dtype=complex)
        else:
            self.u = None
        self.quality = 0
        self.info = info
        self.type = 'Scalar_field_Z'
        self.date = get_date()

    def __str__(self):
        """Represents main data of the atributes."""

        Imin = (np.abs(self.u)**2).min()
        Imax = (np.abs(self.u)**2).max()
        phase_min = (np.angle(self.u)).min() / degrees
        phase_max = (np.angle(self.u)).max() / degrees
        print("{}\n - z:  {},   u:  {}".format(self.type, self.z.shape,
                                               self.u.shape))
        print(" - zmin:       {:2.2f} um,  zmax:      {:2.2f} um,  Dz:   {:2.2f} um".format(
            self.z[0], self.z[-1], self.z[1]-self.z[0]))
        print(" - Imin:       {:2.2f},     Imax:      {:2.2f}".format(
            Imin, Imax))
        print(" - phase_min:  {:2.2f} deg, phase_max: {:2.2f} deg".format(
            phase_min, phase_max))

        print(" - wavelength: {:2.2f} um".format(self.wavelength))
        print(" - date:       {}".format(self.date))
        if self.info != "":
            print(" - info:       {}".format(self.info))
        return ("")

    def __add__(self, other, kind='standard'):
        """Adds two Scalar_field_x. For example two light sources or two masks.

        Parameters:
            other (Scalar_field_X): 2nd field to add
            kind (str): instruction how to add the fields:
                - 'maximum1': mainly for masks. If t3=t1+t2>1 then t3= 1.
                - 'standard': add fields u3=u1+u2 and does nothing.

        Returns:
            Scalar_field_Z: `u3 = u1 + u2`

        TODO: improve
        """

        u3 = Scalar_field_Z(self.z, self.wavelength)

        if kind == 'standard':
            u3.u = self.u + other.u

        elif kind == 'maximum1':
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            f1 = angle(self.u)
            f2 = angle(other.u)
            t3 = t1 + t2
            t3[t3 > 0] = 1.
            u3.u = t3 * exp(1j * (f1 + f2))

        return u3

    def __sub__(self, other):
        """Substract two Scalar_field_x. For example two light sources or two masks.

        Parameters:
            other (Scalar_field_X): field to substract

        Returns:
            Scalar_field_X: `u3 = u1 - u2`

        TODO:
            It can be improved for maks (not having less than 1)
        """

        u3 = Scalar_field_Z(self.z, self.wavelength)
        u3.u = self.u - other.u
        return u3

    def duplicate(self, clear=False):
        """Duplicates the instance"""
        # new_field = Scalar_field_X(self.z, self.wavelength)
        # if clear is False:
        #     new_field.u = self.u
        # return new_field
        new_field = copy.deepcopy(self)
        if clear is True:
            new_field.clear_field()
        return new_field

    def clear_field(self):
        """Removes the field so that self.u = 0. """
        self.u = np.zeros_like(self.u, dtype=complex)

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
        """Load data from a file to a Scalar_field_X.
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

    def cut_resample(self,
                     z_limits='',
                     num_points=[],
                     new_field=False,
                     interp_kind='linear'):
        """Cuts the field to the range (z0,z1). If one of this z0,z1 positions is out of the self.z range it does nothing.
        It is also valid for resampling the field, just write z0,z1 as the limits of self.z

        Parameters:
            z_limits (numpy.array): (z0,z1) - starting and final points to cut, if '' - takes the current limit z[0] and z[-1]
            num_points (int): it resamples z, and u [],'',0,None -> it leave the points as it is
            new_field (bool): if True it returns a new Scalar_field_z
            interp_kind (str): 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'

        Returns:
            (Scalar_field_Z): if new_field is True
        """

        if z_limits == '':
            # used only for resampling
            z0 = self.z[0]
            z1 = self.z[-1]
        else:
            z0, z1 = z_limits

        if z0 < self.z[0]:
            z0 = self.z[0]
        if z1 > self.z[-1]:
            z1 = self.z[-1]

        i_z0, _, _ = nearest(self.z, z0)
        i_z1, _, _ = nearest(self.z, z1)

        if num_points not in ([], '', 0, None):
            z_new = linspace(z0, z1, num_points)
            f_interp_abs = interp1d(self.z,
                                    np.abs(self.u),
                                    kind=interp_kind,
                                    bounds_error=False,
                                    fill_value=0)

            f_interp_phase = interp1d(self.z,
                                      np.imag(self.u),
                                      kind=interp_kind,
                                      bounds_error=False,
                                      fill_value=0)

            u_new_abs = f_interp_abs(z_new)
            u_new_phase = f_interp_phase(z_new)
            u_new = u_new_abs * np.ezp(1j * u_new_phase)

        else:
            i_s = slice(i_z0, i_z1)
            z_new = self.z[i_s]
            u_new = self.u[i_s]

        if new_field is False:
            self.z = z_new
            self.u = u_new
        elif new_field is True:
            field = Scalar_field_Z(z=z_new, wavelength=self.wavelength)
            field.u = u_new
            return field

    def normalize(self, kind='intensity', new_field=False):
        """Normalizes the field so that intensity.max()=1.

        Parameters:
            kind (str): 'intensity, 'amplitude', 'logarithm'... other.. Normalization technique
            new_field (bool): If False the computation goes to self.u. If True a new instance is produced
        Returns
            u (numpy.array): normalized optical field
        """
        u_new = normalize(self.u, kind)

        if new_field is False:
            self.u = u_new
        else:
            field_output = Scalar_field_Z(self.z, self.wavelength)
            field_output.u = u_new
            return field_output

    def intensity(self):
        """Intensity.

        Returns:
            (numpy.array): Intensity
        """

        intensity = (np.abs(self.u)**2)
        return intensity

    def average_intensity(self, verbose=False):
        """Returns the average intensity as: (np.abs(self.u)**2).sum() / num_data

        Parameters:
            verbose (bool): If True it prints the value of the average intensity.

        Returns:
            (float): average intensity.
        """
        average_intensity = (np.abs(self.u)**2).mean()
        if verbose is True:
            print("average intensity={} W/m").format(average_intensity)

        return average_intensity

    def FWHM1D(self, percentage=0.5, remove_background=None, has_draw=False):
        """
        FWHM1D

        remove_background = 'min', 'mean', None
        """

        intensities = np.abs(self.u)**2

        widths = FWHM1D(self.z, intensities, percentage, remove_background, has_draw)
       
        return widths

    def DOF(self, w_factor=np.sqrt(2), w_fixed=0, has_draw=False, verbose=False):
        """Determines Depth-of_focus (DOF) in terms of the width at different distances

        Parameters:

            z (np.array): z positions
            widths (np.array): width at positions z
            w_factor (float): range to determine z where   w = w_factor * w0, being w0 the beam waist
            w_fixed (float): If it is not 0, then it is used as w_min
            has_draw (bool): if True draws the depth of focus
            verbose (bool): if True, prints data

        References:

            B. E. A. Saleh and M. C. Teich, Fundamentals of photonics. john Wiley & sons, 2nd ed. 2007. Eqs (3.1-18) (3.1-22) page 79

        Returns:

            (float): Depth of focus
            (float): beam waist
            (float, float, float): postions (z_min, z_0, z_max) of the depth of focus
        """

        pass


    def draw(self,
             kind='intensity',
             logarithm=False,
             normalize=False,
             cut_value=None,
             z_scale='um',
             unwrap=False,
             filename=''):
        """Draws z field. There are several data from the field that are extracted, depending of 'kind' parameter.

        Parameters:
            kind (str): type of drawing: 'amplitude', 'intensity', 'field', 'phase'
            logarithm (bool): If True, intensity is scaled in logarithm
            normalize (bool): If True, max(intensity)=1
            cut_value (float): If not None, cuts the maximum intensity to this value
            unwrap (bool): If True, unwraps the phase.
            filename (str): if not '' stores drawing in file,
        """

        if self.z is None:
            print('could not draw file: self.z=None')
            return
        if z_scale == 'mm':
            z_drawing = self.z / mm
            zlabel = '$z\,(mm)$'

        else:
            z_drawing = self.z
            zlabel = '$z\,(\mu m)$'

        amplitude, intensity, phase = field_parameters(self.u)

        if unwrap:
            phase = np.unwrap(phase)

        plt.figure()

        if kind == 'intensity':
            y = intensity
        elif kind == 'phase':
            y = phase
        elif kind in ('amplitude', 'field'):
            y = amplitude

        if kind in ('intensity', 'amplitude', 'field'):
            y = normalize_draw(y, logarithm, normalize, cut_value)

        if kind == 'field':
            plt.subplot(211)
            plt.plot(z_drawing, y, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel('$A\,(arb.u.)$')
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])
            plt.ylim(bottom=0)

            plt.subplot(212)
            plt.plot(z_drawing, phase, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel('$phase\,(radians)$')
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])

        elif kind in ('amplitude', 'intensity', 'phase'):
            plt.plot(z_drawing, y, 'k', lw=2)
            plt.xlabel(zlabel)
            plt.ylabel(kind)
            plt.xlim(left=z_drawing[0], right=z_drawing[-1])

        if not filename == '':
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)

        if kind == 'intensity':
            plt.ylim(bottom=0)

        elif kind == 'phase':
            if unwrap == False:
                plt.ylim(-pi, pi)

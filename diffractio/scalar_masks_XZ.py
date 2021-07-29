# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_mask_XZ class for definingn masks. Its parent is Scalar_field_XZ.

The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refraction index XZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic


The magnitude is related to microns: `micron = 1.`


*Class for unidimensional scalar masks*

*Functions*
    * extrude_mask, mask_from_function, mask_from_array, object_by_surfaces
    * image
    * semi_plane, layer, rectangle, slit, sphere, semi_sphere
    * wedge, prism, biprism
    * ronchi_grating, sine_grating
    * probe
    * lens_plane_convergent, lens_convergent, lens_plane_divergent, lens_divergent
    * roughness
"""

from copy import deepcopy

import matplotlib.image as mpimg
import numexpr as ne
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d

from . import degrees, np, plt, sp, um
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_X import Scalar_mask_X
from .utils_math import nearest, nearest2
from .utils_optics import roughness_1D


class Scalar_mask_XZ(Scalar_field_XZ):
    """Class for working with XZ scalar masks.

    Parameters:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        z (numpy.array): linear array wit equidistant positions for z values
        wavelength (float): wavelength of the incident field
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`.
        self.z (numpy.array): linear array wit equidistant positions for z values
        self.wavelength (float): wavelength of the incident field.
        self.u0 (numpy.array): (x) size x - field at the last z position
        self.u (numpy.array): (x,z) complex field
        self.n_background (numpy.array): (x,z) refraction index
        self.info (str): String with info about the simulation
    """

    def __init__(self,
                 x=None,
                 z=None,
                 wavelength=None,
                 n_background=1,
                 info=''):
        """inits a new experiment:
        x: numpy array with x locations
        z: numpy array with z locations
        wavelength: wavelength of light
        n_backgraound: refraction index of background
        info: text to describe the instance of the class
        """
        # print("init de Scalar_mask_XZ")
        super(self.__class__, self).__init__(x, z, wavelength, n_background,
                                             info)

    def extrude_mask(self, t, z0, z1, refraction_index, v_globals={}, angle=0):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Parameters:
            t (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            z0 (float): initial  position of mask
            z1 (float): final position of mask
            refraction_index (float, str): can be a number or a function n(x,z)
        """

        iz0, value, distance = nearest(vector=self.z, number=z0)
        iz1, value, distance = nearest(vector=self.z, number=z1)

        if isinstance(refraction_index, (int, float, complex)):
            n_is_number = True
            # refraction_index = refraction_index * np.ones((iz1 - iz0))
        else:
            n_is_number = False
            v_locals = {'self': self, 'sp': sp, 'degrees': degrees, 'um': um}
            tmp_refraction_index = refraction_index

        for i, index in enumerate(range(iz0, iz1)):
            if n_is_number is False:
                v_locals['z'] = self.z[index]
                v_locals['x'] = self.x

                refraction_index = eval(tmp_refraction_index, v_globals,
                                        v_locals)
                # print("aqui: {}".format(refraction_index))

            self.n[:, index] = refraction_index * (1 - t.u)
            self.n[:, index] = self.n[:, index] + self.n_background * t.u

    def mask_from_function(self,
                           r0,
                           refraction_index,
                           f1,
                           f2,
                           z_sides,
                           angle,
                           v_globals={}):
        """
        Phase mask defined between two surfaces f1 and f1: h(x,z)=f2(x,z)-f1(x,z)

        Parameters:
            r0 (float, float): location of the mask
            refraction_index (float, str): can be a number or a function n(x,z)
            f1 (str): function that delimits the first surface
            f2 (str): function that delimits the second surface
            z_sides (float, float): limiting upper and lower values in z,
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables
        """

        v_locals = {'self': self, 'sp': sp, 'degrees': degrees, 'um': um}

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)
        # Rotacion del square/rectangle
        Xrot, Zrot = self.__rotate__(angle, r0)

        # Transmitancia de los points interiores
        ipasa = (Xrot > z_sides[0]) & (Xrot < z_sides[1]) & (Zrot <
                                                             F2) & (Zrot > F1)
        self.n[ipasa] = refraction_index
        return ipasa

    def mask_from_array(self,
                        r0=(0 * um, 0 * um),
                        refraction_index=1.5,
                        array1=None,
                        array2=None,
                        x_sides=None,
                        angle=0 * degrees,
                        v_globals={},
                        interp_kind='quadratic',
                        has_draw=False):
        """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
        For the definion of f1 and f2 from arrays is performed an interpolation

        Parameters:
            r0 (float, float): location of the mask
            refractdion_index (float, str): can be a number or a function n(x,z)
            array1 (numpy.array): array (x,z) that delimits the first surface
            array2 (numpy.array): array (x,z) that delimits the second surface
            x_sides (float, float): limiting upper and lower values in x,
            angle (float): angle of rotation (radians): TODO -> not working
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            interp_kind: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        """

        x_c, z_c = r0

        f1_interp = interp1d(
            array1[:, 0] + x_c,
            array1[:, 1] + z_c,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array1[0, 1] + z_c,
            assume_sorted=True)

        f2_interp = interp1d(
            array2[:, 0] + x_c,
            array2[:, 1] + z_c,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array2[0, 1] + z_c,
            assume_sorted=True)

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        if has_draw is True:
            plt.figure()
            plt.plot(self.x, F1)
            plt.plot(self.x, F2, 'r')

        Xrot, Zrot = self.__rotate__(angle, r0)

        i_z1, _, _ = nearest2(self.z, F1)
        i_z2, _, _ = nearest2(self.z, F2)
        ipasa = np.zeros_like(self.n, dtype=bool)
        for i, xi in enumerate(self.x):
            #     minor, mayor = min(i_z1[i], i_z2[i]), max(i_z1[i], i_z2[i])
            #     ipasa[i, minor:mayor] = True
            ipasa[i, i_z1[i]:i_z2[i]] = True

        if x_sides is None:
            self.n[ipasa] = refraction_index
            return ipasa

        else:
            ipasa2 = Xrot < x_sides[1]
            ipasa3 = Xrot > x_sides[0]

            self.n[ipasa * ipasa2 * ipasa3] = refraction_index
            return ipasa * ipasa2 * ipasa3

    def mask_from_array_proposal(self,
                                 r0=(0 * um, 0 * um),
                                 refraction_index_substrate=1.5,
                                 refraction_index_mask=None,
                                 array1=None,
                                 array2=None,
                                 x_sides=None,
                                 angle=0 * degrees,
                                 v_globals={},
                                 interp_kind='quadratic',
                                 has_draw=False):
        """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
        For the definion of f1 and f2 from arrays is performed an interpolation

        Parameters:
            r0 (float, float): location of the mask
            refraction_index_mask (float, str): can be a number or a function n(x,z)
            refraction_index_substrate (float, str): can be a number or a function n(x,z)

            array1 (numpy.array): array (x,z) that delimits the first surface
            array2 (numpy.array): array (x,z) that delimits the second surface
            x_sides (float, float): limiting upper and lower values in x,
            angle (float): angle of rotation (radians): TODO -> not working
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            interp_kind: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        """

        x0, z0 = r0

        f1_interp = interp1d(
            array1[:, 0] + x0,
            array1[:, 1] + z0,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array1[0, 1] + z0,
            assume_sorted=True)

        f2_interp = interp1d(
            array2[:, 0] + x0,
            array2[:, 1] + z0,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array2[0, 1] + z0,
            assume_sorted=True)

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        if has_draw is True:
            plt.figure()
            plt.plot(self.x, F1)
            plt.plot(self.x, F2, 'r')

        Xrot, Zrot = self.__rotate__(angle, r0)

        i_z1, _, _ = nearest2(self.z, F1)
        i_z2, _, _ = nearest2(self.z, F2)
        ipasa = np.zeros_like(self.n, dtype=bool)

        for i, xi in enumerate(self.x):
            minor, mayor = min(i_z1[i], i_z2[i]), max(i_z1[i], i_z2[i])
            ipasa[i, minor:mayor] = True

        if refraction_index_mask not in (None, '', []):
            ipasa_substrate = np.zeros_like(self.u)
            z_subst_0 = np.max(F1)
            z_subst_1 = np.min(F2)

            i_z1, _, _ = nearest(self.z, z_subst_0)
            i_z2, _, _ = nearest(self.z, z_subst_1)
            ipasa_substrate[:, i_z1:i_z2] = refraction_index_substrate

        if x_sides is None:
            self.n[ipasa] = refraction_index_mask
            if refraction_index_mask not in (None, '', []):
                ipasa_substrate[:, i_z1:i_z2] = refraction_index_substrate
                self.n[ipasa_substrate] = refraction_index_substrate

            return ipasa

        else:
            ipasa2 = Xrot < x_sides[1]
            ipasa3 = Xrot > x_sides[0]

            self.n[ipasa * ipasa2 * ipasa3] = refraction_index_mask
            self.n[ipasa_substrate] = refraction_index_substrate

            return ipasa * ipasa2 * ipasa3

    def object_by_surfaces(self,
                           rotation_point,
                           refraction_index,
                           Fs,
                           angle,
                           v_globals={},
                           verbose=False):
        """Mask defined by n surfaces given in array Fs={f1, f2, ....}.
        h(x,z)=f1(x,z)*f2(x,z)*....*fn(x,z)


        Parameters:
            rotation_point (float, float): location of the mask
            refraction_index (float, str): can be a number or a function n(x,z)
            Fs (list): condtions as str that will be computed using eval
            array1 (numpy.array): array (x,z) that delimits the second surface
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            verbose (bool): shows data if true
        """

        # Rotacion del square/rectangle
        Xrot, Zrot = self.__rotate__(angle, rotation_point)

        v_locals = {
            'self': self,
            'sp': sp,
            'degrees': degrees,
            'um': um,
            'np': np
        }

        v_locals['Xrot'] = Xrot
        v_locals['Zrot'] = Zrot

        conditions = []
        for fi in Fs:
            # result_condition = eval(fi, v_globals, v_locals)
            try:
                result_condition = ne.evaluate(fi, v_globals, v_locals)
            except:
                result_condition = eval(fi, v_globals, v_locals)

            conditions.append(result_condition)

        # Transmitancia de los puntos interiores
        ipasa = conditions[0]
        for cond in conditions:
            ipasa = ipasa & cond

        if verbose is True:
            print((" n = {}".format(refraction_index)))

        if isinstance(refraction_index, (int, float, complex)):
            self.n[ipasa] = refraction_index
            return ipasa
        else:
            v_locals = {'self': self, 'sp': sp, 'degrees': degrees, 'um': um}
            tmp_refraction_index = refraction_index

            v_locals['X'] = Xrot
            v_locals['Z'] = Zrot

            refraction_index = eval(tmp_refraction_index, v_globals, v_locals)
            self.n[ipasa] = refraction_index[ipasa]
            return ipasa

    def add_surfaces(self,
                     fx,
                     x_sides,
                     refraction_index,
                     min_incr=0.1,
                     angle=0 * degrees):
        """A topography fx is added to one of the faces of object u (self.n).


        Parameters:
            u (Scalar_mask_XZ): topography
            fx (numpy.array, numpy.array):  [x1, fx1], [x2, fx2] array with topography to add
            x_sides (float, float): positions of edges
            refraction_index (float, str): refraction index: number of string
            min_incr (float): minimum variation of refraction index to detect edge.
            angle (float (float, float)): angle and optative rotation angle.
        """
        z0 = self.z
        x0 = self.x
        len_x, len_z = self.n.shape

        # surface detection
        diff1a = np.diff(np.abs(self.n), axis=1)
        diff1a = np.append(diff1a, np.zeros((len_x, 1)), axis=1)

        # cada uno de los lados
        ix_l, iz_l = (diff1a > min_incr).nonzero()
        ix_r, iz_r = (diff1a < -min_incr).nonzero()

        x_lens_l = x0[ix_l]
        h_lens_l = z0[iz_l]

        x_lens_r = x0[ix_r]
        h_lens_r = z0[iz_r]

        fx1, fx2 = fx

        if fx1 is not None:
            x_1, h_1 = fx1  # primera superficie
            h_1_new = np.interp(x_lens_l, x_1, h_1)
            h_lens_l = h_lens_l + h_1_new
        if fx2 is not None:
            x_2, h_2 = fx2  # segunda superficie
            h_2_new = np.interp(x_lens_r, x_2, h_2)
            h_lens_r = h_lens_r + h_2_new

        len_z1 = len(x_lens_l)
        fx1_n = np.concatenate((x_lens_l, h_lens_l)).reshape(2, len_z1).T

        len_z2 = len(x_lens_r)
        fx2_n = np.concatenate((x_lens_r, h_lens_r)).reshape(2, len_z2).T

        perfil_previo = self.borders
        self.clear_refraction_index()
        self.mask_from_array(
            r0=(0 * um, 0 * um),
            refraction_index=refraction_index,
            array1=fx1_n,
            array2=fx2_n,
            x_sides=x_sides,
            angle=0 * degrees,
            interp_kind='linear')

        self.surface_detection()  # bordes nuevos
        perfil_nuevo = self.borders

        return perfil_previo, perfil_nuevo

    def discretize_refraction_index(self, n_layers):
        """Takes a refraction index an discretize it according refraction indexes.

        Parameters:
            n_layers (np.array): array with refraction indexes to discretize
        """
        n = deepcopy(self.n)
        for i, n_layer in enumerate(n_layers[:-1]):
            i_capa = np.bitwise_and(self.n >= n_layers[i],
                                    self.n <= n_layers[i + 1])
            n_central = (n_layers[i] + n_layers[i + 1]) / 2
            n[i_capa] = n_central
        self.n = n
        return n

    def image(self, filename, n_max, n_min, angle=0, invert=False):
        """Converts an image file in an xz-refraction index matrix.
        If the image is gray-scale the refraction index is gradual betwee n_min and n_max.
        If the image is color, we get the first Red frame

        Parameters:
            filename (str): filename of the image
            n_max (float): maximum refraction index
            n_min (float): minimum refraction index
            angle (float): angle to rotate the image in radians
            invert (bool): if True the image is inverted

        TODO:
            Now it is only possible that image size is equal to XZ, change using interpolation
            Rotation position
        """

        image3D = mpimg.imread(filename)
        if len(image3D.shape) > 2:
            image = image3D[:, :, 0]
        else:
            image = image3D
        """
        lengthImage = True
        if lengthImage is False:
            length = (len(xsampling), len(ysampling))
            image = np.resize(image,length)

        if lengthImage is True:
            # length = im.size
            length = image.shape
            self.x = linspace(self.x[0], self.x[-1], length[0])
            self.y = linspace(self.y[0], self.y[-1], length[1])
            # self.X, self.Y = meshgrid(self.x, self.y)
            X, Y = meshgrid(self.x, self.y)
        """
        # angle is in degrees
        image = ndimage.rotate(image, angle * 180 / np.pi, reshape=False)
        image = np.array(image)
        image = (image - image.min()) / (image.max() - image.min())

        if invert is False:
            image = image.max() - image
        self.n = n_min + image * (n_max - n_min)

    def dots(self, positions, refraction_index=1):
        """Generates 1 or several point masks at positions r0

        Parameters:
            positions (float, float) or (np.array, np.array): (x,z) point or points where mask is 1
            refraction_index (float): refraction index
        """
        x0, z0 = positions
        n = np.zeros_like(self.X)

        if type(positions[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_z0, _, _ = nearest(self.z, z0)
            n[i_x0, i_z0] = refraction_index
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_z0s, _, _ = nearest2(self.z, z0)
            for (i_x0, i_z0) in zip(i_x0s, i_z0s):
                n[i_x0, i_z0] = refraction_index

        self.n = n
        return self

    def semi_plane(self, r0, refraction_index, angle=0, rotation_point=None):
        """Inserts a semi-sphere in background (x>x0). If it is something else previous, it is removed.

        Parameters:
            r0=(x0,z0) (float,float): Location of the same plane.
            refraction_index (float, str): refraction index.
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0  # DUDA

        cond1 = "Zrot>{}".format(z0)

        Fs = [cond1]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def layer(self, r0, depth, refraction_index, angle, rotation_point=None):
        """ Insert a layer. If it is something else previous, it is removed.

        Parameters:
        r0 (float, float): (x0,z0) Location of the same plane, for example (0 * um, 20 * um)
        depth (float): depth of the layer
        refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
        angle (float): angle of rotation of the semi-plane, in radians
        rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Zrot<{}".format(z0 + depth)

        Fs = [cond1, cond2]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def rectangle(self,
                  r0,
                  size,
                  refraction_index,
                  angle=0 * degrees,
                  rotation_point=None):
        """ Insert a rectangle in background. Something previous, is removed.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            size (float, float): x,z size of the rectangle
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(size, (float, int, complex)):
            sizex, sizez = size, size
        else:
            sizex, sizez = size

        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        # Definicion del square/rectangle
        xmin = x0 - sizex / 2
        xmax = x0 + sizex / 2
        zmin = z0 - sizez / 2
        zmax = z0 + sizez / 2

        # Transmitancia de los points interiores

        cond1 = "Xrot<{}".format(xmax)
        cond2 = "Xrot>{}".format(xmin)
        cond3 = "Zrot<{}".format(zmax)
        cond4 = "Zrot>{}".format(zmin)

        Fs = [cond1, cond2, cond3, cond4]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={'np': np})

        return ipasa

    def slit(self,
             r0,
             aperture,
             depth,
             refraction_index,
             refraction_index_center='',
             angle=0,
             rotation_point=None):
        """ Insert a slit in background.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            aperture (float): length of the opened part of the slit
            depth (float): depth of the slit
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            refraction_index_center (float, str?): refraction index of center
                if refraction_index_center='', [], 0 then we copy what it was previously at aperture
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        n_back = deepcopy(self.n)

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Zrot<{}".format(z0 + depth)
        cond3 = "Xrot<{}".format(x0 + aperture / 2)
        cond4 = "Xrot>{}".format(x0 - aperture / 2)

        Fs1 = [cond1, cond2]

        ipasa_slit = self.object_by_surfaces(
            r0, refraction_index, Fs1, angle, v_globals={})

        Fs2 = [cond1, cond2, cond3, cond4]
        if refraction_index_center not in ('', [], 0):
            ipasa = self.object_by_surfaces(
                r0, refraction_index_center, Fs2, angle, v_globals={})
        elif refraction_index_center in ('', [], 0):
            ipasa = self.object_by_surfaces(r0, 1, Fs2, angle, v_globals={})
            self.n[ipasa] = n_back[ipasa]
        return ipasa_slit != ipasa

    def sphere(self,
               r0,
               radius,
               refraction_index,
               angle=0,
               rotation_point=None):
        """ Insert a sphere in background.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            radius (float, float): radius x,y of the sphere (ellipsoid)
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusz = (radius, radius)
        else:
            radiusx, radiusz = radius

        cond = "(Xrot - {})**2 / {}**2 + (Zrot- {})**2 / {}**2 < 1".format(
            x0, radiusx, z0, radiusz)

        Fs = [cond]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def semi_sphere(self,
                    r0,
                    radius,
                    refraction_index,
                    angle=0,
                    rotation_point=None):
        """ Insert a semi_sphere in background.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            radius (float, float): radius x,y of the sphere (ellipsoid)
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusz = (radius, radius)
        else:
            radiusx, radiusz = radius

        cond1 = "Zrot>{}".format(z0)
        cond2 = "(Xrot - {})**2 / {}**2 + (Zrot- {})**2 / {}**2 < 1".format(
            x0, radiusx, z0, radiusz)

        Fs = [cond1, cond2]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def lens_plane_convergent(self,
                              r0,
                              aperture,
                              radius,
                              thickness,
                              refraction_index,
                              angle=0,
                              rotation_point=None,
                              mask=0):
        """Insert a plane-convergent lens in background-

        Parameters:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um)
              for plane-convergent z0 is the location of the plane
              for convergent-plane (angle =180*degrees) the thickness has to be
                  added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float): radius of the curved surface
            thickness (float): thickness at the center of the lens
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            mask (array, str):  (mask_depth, refraction_index) or False.
                It masks the field outer the lens using a slit with depth = mask_depth
            rotation_point (float, float). Rotation point

        Returns:
            (float): geometrical focal distance
            (numpy.array): ipasa, indexes [iz,ix] of lens
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        z_plane = z0
        z_center_lens = z_plane + thickness - radius
        if mask is False:
            mask_depth = 0
            mask_refraction_index = 1 - 0.1j
        else:
            mask_depth, mask_refraction_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_plane = "Zrot>{}".format(z_plane)
        cond_radius = "(Xrot - {})**2 +(Zrot -{})**2 <{}**2".format(
            x0, z_center_lens, radius)
        Fs = [cond_aperture1, cond_aperture2, cond_plane, cond_radius]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refraction_index=mask_refraction_index,
                refraction_index_center='',
                angle=angle,
                rotation_point=rotation_point)
        focus = radius / (refraction_index - 1)
        return focus, ipasa

    def lens_convergent(self,
                        r0,
                        aperture,
                        radius,
                        thickness,
                        refraction_index,
                        angle=0,
                        rotation_point=None,
                        mask=0):
        """Inserts a convergent lens in background.

        Parameters:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float, float): (radius1,radius2) radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float): rotation point.
            mask (array, str):  (mask_depth, refraction_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

        Returns:
            (float): geometrical focal distance
            (numpy.array): ipasa, indexes [iz,ix] of lens
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        radius1, radius2 = radius
        z_center_lens1 = z0 + radius1
        z_center_lens2 = z0 + radius2 + thickness

        # print(("z={},{}".format(z_center_lens1, z_center_lens2)))
        # print(("r={},{}".format(radius1, radius2)))

        if mask is False:
            mask_depth = 0
            mask_refraction_index = 1 - 0.1j
        else:
            mask_depth, mask_refraction_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_radius1 = "(Xrot - {})**2 +(Zrot -{})**2 <({})**2".format(
            x0, z_center_lens1, radius1)
        cond_radius2 = "(Xrot - {})**2 +(Zrot -{})**2 <({})**2".format(
            x0, z_center_lens2, -radius2)

        Fs = [cond_aperture1, cond_aperture2, cond_radius1, cond_radius2]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refraction_index=mask_refraction_index,
                refraction_index_center='',
                angle=angle)
        focus_1 = (refraction_index - 1) * (
            1 / radius1 - 1 / radius2) - thickness / (radius1 * radius2)
        return 1 / focus_1, ipasa

    def lens_plane_divergent(self,
                             r0,
                             aperture,
                             radius,
                             thickness,
                             refraction_index,
                             angle=0,
                             rotation_point=None,
                             mask=False):
        """Insert a plane-divergent lens in background.

        Parameters:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float): radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            mask (array, str):  (mask_depth, refraction_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

        Returns:
            (float): geometrical focal distance
            (numpy.array): ipasa, indexes [iz,ix] of lens
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        z_center_lens = z0 + thickness + radius
        if mask is False:
            mask_depth = 0
            mask_refraction_index = 1 - 0.1j
        else:
            mask_depth, mask_refraction_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_plane = "Zrot>{}".format(z0)
        cond_radius = "(Xrot - {})**2 +(Zrot -{})**2 >({})**2".format(
            x0, z_center_lens, radius)
        cond_right = "Zrot<{}".format(z_center_lens)
        Fs = [
            cond_aperture1, cond_aperture2, cond_plane, cond_radius, cond_right
        ]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refraction_index=mask_refraction_index,
                refraction_index_center='',
                angle=angle,
                rotation_point=rotation_point)
        focus = radius / (refraction_index - 1)
        return focus, ipasa

    def lens_divergent(self,
                       r0,
                       aperture,
                       radius,
                       thickness,
                       refraction_index,
                       angle=0,
                       rotation_point=None,
                       mask=0):
        """Insert a  divergent lens in background.

        Parameters:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float, float): (radius1, radius2) radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float): rotation point
            mask (array, str):  (mask_depth, refraction_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

        Returns:
            (float): geometrical focal distance
            (numpy.array): ipasa, indexes [iz,ix] of lens
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        radius1, radius2 = radius
        z_center_lens1 = z0 + radius1
        z_center_lens2 = z0 + radius2 + thickness

        if mask is False:
            mask_depth = 0
            mask_refraction_index = 1 - 0.1j
        else:
            mask_depth, mask_refraction_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_radius1 = "(Xrot - {})**2 +(Zrot -{})**2>({})**2".format(
            x0, z_center_lens1, radius1)
        cond_radius2 = "(Xrot - {})**2 +(Zrot -{})**2 >({})**2".format(
            x0, z_center_lens2, -radius2)
        cond_right = "Zrot>{}".format(z_center_lens1)
        cond_left = "Zrot<{}".format(z_center_lens2)

        Fs = [
            cond_aperture1, cond_aperture2, cond_radius1, cond_radius2,
            cond_right, cond_left
        ]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refraction_index=mask_refraction_index,
                refraction_index_center='',
                angle=angle,
                rotation_point=rotation_point)
        focus_1 = (refraction_index - 1) * (
            1 / radius1 - 1 / radius2) - thickness / (radius1 * radius2)
        return 1 / focus_1, ipasa

    def aspheric_surface_z(self, r0, refraction_index, cx, Qx, a2, a3, a4,
                           side, angle):
        """Define an aspheric surface

        Parameters:
            r0 (float, float): (x0,z0) position of apex
            refraction_index (float, str): refraction index
            cx (float): curvature
            Qx (float): Conic constant
            side (str): 'left', 'right'

        Returns:
            numpy.array   : Bool array with positions inside the surface
        """

        x0, z0 = r0

        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        if side == 'right':
            sign = '>'
        elif side == 'left':
            sign = '<'
        else:
            print("possible error in aspheric")

        params = dict(
            x0=x0, z0=z0, cx=cx, Qx=Qx, a2=a2, a3=a3, a4=a4, sign=sign)

        cond = "Zrot{sign}{z0}+{cx}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx})*{cx}**2*(Xrot-{x0})**2+{a2}*(Xrot-{x0})**4+{a3}*(Xrot-{x0})**6)+{a4}*(Xrot-{x0})**8)".format(
            **params)

        Fs = [cond]
        v_globals = {'self': self, 'np': np, 'degrees': degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals=v_globals)
        return ipasa

    def aspheric_lens(self,
                      r0,
                      angle,
                      refraction_index,
                      cx,
                      Qx,
                      depth,
                      size,
                      a2=(0, 0),
                      a3=(0, 0),
                      a4=(0, 0)):
        """Define an aspheric surface as defined in Gomez-Pedrero.

        Parameters:
            r0 (float, float): position x,z of lens
            angle (float): rotation angle of lens + r0_rot
            cx (float, float): curvature
            Qx (float, float): Conic constant
            depth  (float, float): distance of the apex
            size (float): diameter of lens

        Returns:
            numpy.array   : Bool array with positions inside the surface
        """
        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        cx1, cx2 = cx
        Qx1, Qx2 = Qx
        a21, a22 = a2
        a31, a32 = a3
        a41, a42 = a4
        side1, side2 = 'left', 'right'

        if side1 == 'right':
            sign1 = '<'
        else:
            sign1 = '>'

        if side2 == 'right':
            sign2 = '<'
        else:
            sign2 = '>'

        params = dict(
            cx1=cx1,
            Qx1=Qx1,
            cx2=cx2,
            Qx2=Qx2,
            x0=x0,
            a21=a21,
            a22=a22,
            a31=a31,
            a32=a32,
            a41=a41,
            a42=a42,
            d1=z0,
            d2=z0 + depth,
            sign1=sign1,
            sign2=sign2)

        cond1 = "Zrot{sign1}{d1}+{cx1}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx1})*{cx1}**2*(Xrot-{x0})**2+{a21}*(Xrot-{x0})**4+{a31}*(Xrot-{x0})**6)+{a41}*(Xrot-{x0})**8)".format(
            **params)

        cond2 = "Zrot{sign2}{d2}+{cx2}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx2})*{cx2}**2*(Xrot-{x0})**2+{a22}*(Xrot-{x0})**4+{a32}*(Xrot-{x0})**6)+{a42}*(Xrot-{x0})**8)".format(
            **params)

        cond3 = "(Xrot-{})<{}".format(x0, size / 2)
        cond4 = "(Xrot-{})>{}".format(x0, -size / 2)

        cond5 = "Zrot > {}".format(z0 - depth)
        cond6 = "Zrot < {}".format(z0 + depth)

        Fs = [cond1, cond2, cond3, cond4, cond5, cond6]
        v_globals = {'self': self, 'np': np, 'degrees': degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals=v_globals)

        return ipasa, Fs

    def wedge(self,
              r0,
              length,
              refraction_index,
              angle_wedge,
              angle=0,
              rotation_point=None):
        """ Insert a wedge pointing towards the light beam

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle_wedge (float), angle of the wedge in radians
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Xrot>{}".format(x0)
        cond2 = "Zrot<({}+{})".format(z0, length)
        cond3 = "(Xrot-{})<{}*(Zrot-{})".format(x0, np.tan(angle_wedge), z0)
        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def prism(self,
              r0,
              length,
              refraction_index,
              angle_prism,
              angle=0,
              rotation_point=None):
        """Similar to wedge but the use is different. Also the angle is usually different. One of the sides is paralel to x=x0

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle_prism (float), angle of the prism in radians
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Xrot>{}".format(x0)
        cond2 = "Zrot-({})>{}*(Xrot-{})".format(z0, np.tan(angle_prism / 2),
                                                x0)
        cond3 = "Zrot-({})<{}*(Xrot-{})".format(
            z0 + length, np.tan(np.pi - angle_prism / 2), x0)

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def biprism(self, r0, length, height, refraction_index, angle=0):
        """Fresnel biprism.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            height (float): height of biprism
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        vuelta = 1
        if vuelta == 1:
            cond1 = "Zrot>{}".format(z0)
            cond2 = "Zrot-({})<{}*(Xrot-{})".format(z0 + height,
                                                    -2 * height / length, x0)
            cond3 = "Zrot-({})<{}*(Xrot-{})".format(z0 + height,
                                                    2 * height / length, x0)
        else:
            cond1 = "Zrot<{}".format(z0)
            cond2 = "Zrot-({})>{}*(Xrot-{})".format(z0 - height,
                                                    -2 * height / length, x0)
            cond3 = "Zrot-({})>{}*(Xrot-{})".format(z0 - height,
                                                    +2 * height / length, x0)

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals={})
        return ipasa

    def ronchi_grating(self, r0, period, fill_factor, length, height, Dx,
                       refraction_index, heigth_substrate,
                       refraction_index_substrate, angle):
        """Insert a ronchi grating in background.

        Parameters:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            length (float): length of the grating
            height (float): height of the grating
            Dx (float): displacement of grating with respect x=0
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            heigth_substrate (float): height of the substrate
            refraction_index_substrate (float, str): refraction index of substrate , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0

        Xrot, Zrot = self.__rotate__(angle, r0)

        t0 = Scalar_mask_X(x=self.x, wavelength=self.wavelength)
        t0.ronchi_grating(x0=Dx, period=period, fill_factor=fill_factor)

        self.extrude_mask(
            t=t0,
            z0=z0 + heigth_substrate / 2,
            z1=z0 + heigth_substrate / 2 + height,
            refraction_index=refraction_index,
            angle=angle)

        if heigth_substrate > 0:
            self.rectangle(r0, (length, heigth_substrate),
                           refraction_index_substrate, angle)
        self.slit(
            r0=(x0, z0 + heigth_substrate / 2),
            aperture=length,
            depth=height,
            refraction_index=self.n_background,
            refraction_index_center='',
            angle=angle)

    def sine_grating(self,
                     period,
                     heigth_sine,
                     heigth_substrate,
                     r0,
                     length,
                     Dx,
                     refraction_index,
                     angle=0):
        """Insert a sine grating in background.

        Parameters:
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            heigth_sine (float): height of the grating
            heigth_substrate (float): height of the substrate
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the grating
            Dx (float): displacement of grating with respect x=0
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        Xrot, Zrot = self.__rotate__(angle, r0)

        c1 = Zrot < z0 + heigth_substrate - heigth_sine / 2 + heigth_sine / 2 * np.cos(
            2 * np.pi * (Xrot - x0 - Dx) / period)
        c2 = Zrot > z0
        conditionZ = c1 * c2  # no es sin, es square
        conditionX = (Xrot > x0 - length / 2) * (Xrot < x0 + length / 2)
        ipasa = conditionZ * conditionX
        self.n[ipasa] = refraction_index
        return ipasa

    def probe(self, r0, base, length, refraction_index, angle):
        """Probe with a sinusoidal shape.

        Parameters:
            r0 (float, float): (x0,z0) position of the center of base, for example (0 * um, 20 * um)
            base (float): base of the probe
            length (float): length of the graprobeing
            Dx (float): displacement of grating with respect x=0
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        if isinstance(angle, (float, int, complex)):
            rotation_point = r0
        else:
            angle, rotation_point = angle

        cond1 = "Zrot<{}+{}/2*np.cos(2*np.pi*Xrot/{})".format(
            length - z0, length, base)
        cond2 = "Xrot<{}".format(x0 + base / 2)
        cond3 = "Xrot>{}".format(x0 - base / 2)
        cond4 = "Zrot>{}".format(z0)
        Fs = [cond1, cond2, cond3, cond4]
        v_globals = {'self': self, 'np': np, 'degrees': degrees, 'um': um}
        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, angle, v_globals=v_globals)
        return ipasa

    def rough_sheet(self,
                    r0,
                    size,
                    t,
                    s,
                    refraction_index,
                    angle,
                    rotation_point=None):
        """Sheet with one of the surface rough.

        Parameters:
            r0 (float, float):(x0,z0) Location of sphere, for example (0 * um, 20 * um)
            size (float, float): (sizex, sizez) size of the sheet
            s (float): std roughness
            t (float): correlation length of roughness
            refraction_index (float, str): refraction index
            angle (float): angle
            rotation_point (float, float): rotation point

        Returns:
            (numpy.array): ipasa, indexes [iz,ix] of lens

        References:
            According to Ogilvy p.224
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        if isinstance(size, (float, int, complex)):
            sizex, sizez = size, size
        else:
            sizex, sizez = size

        k = 2 * np.pi / self.wavelength
        xmin = x0 - sizex / 2
        xmax = x0 + sizex / 2

        # I do not want to touch the previous

        n_back = deepcopy(self.n)

        h_corr = roughness_1D(self.x, t, s)

        fx = h_corr / (k * (refraction_index - 1))  # heights

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Xrot<{}".format(xmax)
        cond3 = "Xrot>{}".format(xmin)

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refraction_index, Fs, 0, v_globals={})

        i_z, _, _ = nearest2(self.z, z0 + sizez - fx)
        i_final = len(self.z)
        for i in range(len(self.x)):
            self.n[i, i_z[i]:i_final] = n_back[i, i_z[i]:i_final]

        self.rotate_field(
            angle, rotation_point, n_background=self.n_background)
        return ipasa

#!/usr/bin/env python3

"""Generates Scalar_mask_XZ class for definingn masks. Its parent is Scalar_field_XZ.

The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refractive index XZ
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

# flake8: noqa


from copy import deepcopy

import matplotlib.image as mpimg
import numexpr as ne
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d

from .utils_typing import npt, Any, NDArray, floating, NDArrayFloat, NDArrayComplex

from . import degrees, np, plt, sp, um
from .scalar_fields_XZ import Scalar_field_XZ
from .scalar_masks_X import Scalar_mask_X
from .utils_math import nearest, nearest2
from .utils_optics import roughness_1D
from .utils_dxf import load_dxf


class Scalar_mask_XZ(Scalar_field_XZ):
    """Class for working with XZ scalar masks.

    Args:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        z (numpy.array): linear array wit equidistant positions for z values
        wavelength (float): wavelength of the incident field
        n_background (float): refractive index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`.
        self.z (numpy.array): linear array wit equidistant positions for z values
        self.wavelength (float): wavelength of the incident field.
        self.u0 (numpy.array): (x) size x - field at the last z position
        self.u (numpy.array): (x,z) complex field
        self.n_background (numpy.array): (x,z) refractive index
        self.info (str): String with info about the simulation
    """

    def __init__(self, x: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        """inits a new experiment:
        x: numpy array with x locations
        z: numpy array with z locations
        wavelength: wavelength of light
        n_backgraound: refractive index of background
        info: text to describe the instance of the class"""
        super().__init__(x, z, wavelength, n_background, info)
        self.type = "Scalar_mask_XZ"

    def extrude_mask(self, t, z0: floating, z1: floating, refractive_index: floating,
                     angle: floating = 0., v_globals: dict = {}):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Args:
            t (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            z0 (float): initial  position of mask
            z1 (float): final position of mask
            refractive_index (float, str): can be a number or a function n(x,z)
        """

        iz0, value, distance = nearest(vector=self.z, number=z0)
        iz1, value, distance = nearest(vector=self.z, number=z1)

        if isinstance(refractive_index, (int, float, complex)):
            n_is_number = True
            # refractive_index = refractive_index * np.ones((iz1 - iz0))
        else:
            n_is_number = False
            v_locals = {"self": self, "sp": sp, "degrees": degrees, "um": um}
            tmp_refractive_index = refractive_index

        for i, index in enumerate(range(iz0, iz1)):
            if n_is_number is False:
                v_locals["z"] = self.z[index]
                v_locals["x"] = self.x

                refractive_index = eval(
                    tmp_refractive_index, v_globals, v_locals)
            self.n = self.n.astype(complex)
            self.n[index, :] = refractive_index * (1 - t.u)
            self.n[index, :] = self.n[index, :] + self.n_background * t.u
            # self.n = refractive_index

    def mask_from_function(
            self, r0: list[floating], refractive_index: floating | str, f1, f2, z_sides: list[floating],
            angle: floating, v_globals: dict = {}):
        """
        Phase mask defined between two surfaces f1 and f1: h(x,z)=f2(x,z)-f1(x,z)

        Args:
            r0 (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            f1 (str): function that delimits the first surface
            f2 (str): function that delimits the second surface
            z_sides (float, float): limiting upper and lower values in z,
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables
        """

        v_locals = {"self": self, "sp": sp, "degrees": degrees, "um": um}

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)
        # Rotacion del square/rectangle
        Xrot, Zrot = self.__rotate__(angle, r0)

        # Transmitancia de los points interiores
        ipasa = (Xrot > z_sides[0]) & (
            Xrot < z_sides[1]) & (Zrot < F2) & (Zrot > F1)
        self.n[ipasa] = refractive_index
        return ipasa

    def mask_from_array(
        self,
        r0=(0 * um, 0 * um),
        refractive_index=1.5,
        array1: NDArrayFloat | None = None,
        array2: NDArrayFloat | None = None,
        x_sides: list[float, float] | None = None,
        angle: floating = 0 * degrees,
        v_globals: dict = {},
        interp_kind: str = "quadratic",
        has_draw: bool = False,
    ):
        """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
        For the definion of f1 and f2 from arrays is performed an interpolation

        Args:
            r0 (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
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
            assume_sorted=True,
        )

        f2_interp = interp1d(
            array2[:, 0] + x_c,
            array2[:, 1] + z_c,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array2[0, 1] + z_c,
            assume_sorted=True,
        )

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        if has_draw is True:
            plt.figure()
            plt.plot(self.x, F1)
            plt.plot(self.x, F2, "r")

        Xrot, Zrot = self.__rotate__(angle, r0)

        i_z1, _, _ = nearest2(self.z, F1)
        i_z2, _, _ = nearest2(self.z, F2)
        ipasa = np.zeros_like(self.n, dtype=bool)
        for i, xi in enumerate(self.x):
            #     minor, mayor = min(i_z1[i], i_z2[i]), max(i_z1[i], i_z2[i])
            #     ipasa[i, minor:mayor] = True
            ipasa[i, i_z1[i]: i_z2[i]] = True

        if x_sides is None:
            self.n[ipasa] = refractive_index
            return ipasa

        else:
            ipasa2 = Xrot < x_sides[1]
            ipasa3 = Xrot > x_sides[0]

            self.n[ipasa * ipasa2 * ipasa3] = refractive_index
            return ipasa * ipasa2 * ipasa3

    def mask_from_array_proposal(
        self,
        r0: list[float, float] = (0 * um, 0 * um),
        refractive_index_substrate: float | floating = 1.5,
        refractive_index_mask: float | floating = None,
        array1: NDArrayFloat | float = None,
        array2: NDArrayFloat | float = None,
        x_sides: list[float, float] = None,
        angle: floating = 0.,
        v_globals: dict = {},
        interp_kind: str = "quadratic",
        has_draw: bool = False,
    ):
        """Mask defined between two surfaces given by arrays (x,z): h(x,z)=f2(x,z)-f1(x,z).
        For the definion of f1 and f2 from arrays is performed an interpolation

        Args:
            r0 (float, float): location of the mask
            refractive_index_mask (float, str): can be a number or a function n(x,z)
            refractive_index_substrate (float, str): can be a number or a function n(x,z)

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
            assume_sorted=True,
        )

        f2_interp = interp1d(
            array2[:, 0] + x0,
            array2[:, 1] + z0,
            kind=interp_kind,
            bounds_error=False,
            fill_value=array2[0, 1] + z0,
            assume_sorted=True,
        )

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        if has_draw is True:
            plt.figure()
            plt.plot(self.x, F1)
            plt.plot(self.x, F2, "r")

        Xrot, Zrot = self.__rotate__(angle, r0)

        i_z1, _, _ = nearest2(self.z, F1)
        i_z2, _, _ = nearest2(self.z, F2)
        ipasa = np.zeros_like(self.n, dtype=bool)

        for i, xi in enumerate(self.x):
            minor, mayor = min(i_z1[i], i_z2[i]), max(i_z1[i], i_z2[i])
            ipasa[i, minor:mayor] = True

        if refractive_index_mask not in (None, "", []):
            ipasa_substrate = np.zeros_like(self.u)
            z_subst_0 = np.max(F1)
            z_subst_1 = np.min(F2)

            i_z1, _, _ = nearest(self.z, z_subst_0)
            i_z2, _, _ = nearest(self.z, z_subst_1)
            ipasa_substrate[:, i_z1:i_z2] = refractive_index_substrate

        if x_sides is None:
            self.n[ipasa] = refractive_index_mask
            if refractive_index_mask not in (None, "", []):
                ipasa_substrate[:, i_z1:i_z2] = refractive_index_substrate
                self.n[ipasa_substrate] = refractive_index_substrate

            return ipasa

        else:
            ipasa2 = Xrot < x_sides[1]
            ipasa3 = Xrot > x_sides[0]

            self.n[ipasa * ipasa2 * ipasa3] = refractive_index_mask
            self.n[ipasa_substrate] = refractive_index_substrate

            return ipasa * ipasa2 * ipasa3

    def object_by_surfaces(
        self, rotation_point: list[float, float], refractive_index: float | str,
        Fs: list, angle: floating, v_globals: dict = {}, verbose: bool = False
    ):
        """Mask defined by n surfaces given in array Fs={f1, f2, ....}.
        h(x,z)=f1(x,z)*f2(x,z)*....*fn(x,z)


        Args:
            rotation_point (float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x,z)
            Fs (list): condtions as str that will be computed using eval
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables -> TODO perphaps it is not necessary
            verbose (bool): shows data if true
        """

        # Rotacion del square/rectangle
        Xrot, Zrot = self.__rotate__(angle, rotation_point)

        v_locals = {"self": self, "sp": sp,
                    "degrees": degrees, "um": um, "np": np}

        v_locals["Xrot"] = Xrot
        v_locals["Zrot"] = Zrot

        conditions = []
        for fi in Fs:
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
            print(("n = {}".format(refractive_index)))

        if isinstance(refractive_index, (int, float, complex)):
            self.n[ipasa] = refractive_index
            return ipasa
        else:
            v_locals = {"self": self, "sp": sp, "degrees": degrees, "um": um}
            tmp_refractive_index = refractive_index

            v_locals["X"] = Xrot
            v_locals["Z"] = Zrot

            refractive_index = eval(tmp_refractive_index, v_globals, v_locals)
            self.n[ipasa] = refractive_index[ipasa]
            return ipasa

    def add_surfaces(
            self, fx: list[NDArrayFloat, NDArrayFloat], x_sides: list[float, float],
            refractive_index: floating | str, min_incr: floating = 0.1, angle: floating = 0.):
        """A topography fx is added to one of the faces of object u (self.n).

        Args:
            u (Scalar_mask_XZ): topography
            fx (numpy.array, numpy.array):  [x1, fx1], [x2, fx2] array with topography to add
            x_sides (float, float): positions of edges
            refractive_index (float, str): refractive index: number of string
            min_incr (float): minimum variation of refractive index to detect edge.
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
        self.clear_refractive_index()
        self.mask_from_array(
            r0=(0 * um, 0 * um),
            refractive_index=refractive_index,
            array1=fx1_n,
            array2=fx2_n,
            x_sides=x_sides,
            angle=0 * degrees,
            interp_kind="linear",
        )

        self.surface_detection()  # bordes nuevos
        perfil_nuevo = self.borders

        return perfil_previo, perfil_nuevo

    def discretize_refractive_index(
        self, num_layers: list[int, int] | None = None, n_layers: NDArrayComplex | complex = None,
        new_field: bool = False
    ):
        """Takes a refractive index an discretize it according refractive indexes.

        Args:
            num_layers ((int,int), optional): number of layers (for real and imaginary parts), without counting background. Defaults to None.
                By default, both parameters are None, but one of then must be filled. If both are present, num_layers is considered
            n_layers (np.array, optional): array with refractive indexes to discretize. Defaults to None.
            new_field (bool): If True, it generates a new field.

        Returns:
            (np.array): refractive indexes selected.
        """

        def _discretize_(refractive_index: NDArrayComplex, n_layers: int):
            """internal function to discretize the refractive index

            Args:
                refractive_index (NDArrayComplex): refractive index
                n_layers (int): number of layers

            Returns:
                NDArrayComplex: discretized refractive index
            """
            n_new = np.zeros_like(refractive_index, dtype=float)

            i_n, _, _ = nearest2(n_layers, refractive_index)
            i_n = i_n.reshape(refractive_index.shape)

            for i, n in enumerate(n_layers):
                n_new[i_n == i] = n_layers[i]

            return n_new

        if num_layers is not None:
            if isinstance(num_layers, int):
                num_layers = (num_layers, 1)

        n_real = np.around(self.n.real, 4)
        kappa = np.around(self.n.imag, 4)

        if num_layers[0] is not None:
            if num_layers[0] > 1:
                repeated_values = np.unique(n_real)
                repeated_values = np.delete(
                    repeated_values, np.where(
                        repeated_values == self.n_background)
                )

                n_min, n_max = repeated_values.min(), repeated_values.max()
                n_layers = np.linspace(n_min, n_max, num_layers[0])
                n_layers = np.append(n_layers, self.n_background)
                n_layers = np.unique(n_layers)
                n_new = _discretize_(n_real, n_layers)
            else:
                n_new = n_real
        else:
            n_new = n_real

        if num_layers[1] is not None:
            if num_layers[1] > 1:
                repeated_values = np.unique(kappa)

                k_min, k_max = repeated_values.min(), repeated_values.max()
                k_layers = np.linspace(k_min, k_max, num_layers[1])
                k_layers = np.unique(k_layers)
                k_new = _discretize_(kappa, k_layers)
            else:
                k_new = np.zeros_like(kappa)
                k_new[kappa > kappa.max() / 2] = kappa.max()
        else:
            k_new = kappa

        if new_field is True:
            t_new = self.duplicate()
            t_new.u = np.zeros_like(t_new.u)
            t_new.n = n_new + 1j * k_new
            return t_new
        else:
            self.n = n_new + 1j * k_new

    def image(self, filename: str, n_max: floating, n_min: floating, angle: floating = 0.,
              invert: bool = False):
        """Converts an image file in an xz-refractive index matrix.
        If the image is gray-scale the refractive index is gradual betwee n_min and n_max.
        If the image is color, we get the first Red frame

        Args:
            filename (str): filename of the image
            n_max (float): maximum refractive index
            n_min (float): minimum refractive index
            angle (float): angle to rotate the image in radians
            invert (bool): if True the image is inverted

        TODO: Now it is only possible that image size is equal to XZ, change using interpolation
            Rotation position
        """

        image3D = mpimg.imread(filename)
        if len(image3D.shape) > 2:
            image = image3D[:, :, 0]
        else:
            image = image3D

        # angle is in degrees
        image = ndimage.rotate(image, angle * 180 / np.pi, reshape=False)
        image = np.array(image)
        image = (image - image.min()) / (image.max() - image.min())

        if invert is False:
            image = image.max() - image
        self.n = n_min + image * (n_max - n_min)

    def dxf(self, filename_dxf: str, n_max: floating, n_min: floating, num_pixels: list[int, int] | None = None, extent: list[float] | None = None,
            units: str = 'mm', invert: bool = False, filename_png: str = 'new.png', has_draw: bool = False, verbose: bool = False):
        """Loads a dxf file. Internally it has the extension of the drawing, so it is not required to generate x,y spaces. It is possible with extent, but then the file is scaled. Warning: Dxf files are usually in mm. and diffractio works in um. To generate .u, a temporal .png file is generated. 

        Args:
            filename_dxf (str): DXF filename .dxf
            num_pixels (list[int, int] | None, optional): If . Defaults to None.
            extent (_type_, optional): _description_. Defaults to None.
            units (str, optional): _description_. Defaults to 'mm'.
            invert (bool, optional): _description_. Defaults to False.
            filename_png (str, optional): _description_. Defaults to 'new.png'.
            has_draw (bool, optional): _description_. Defaults to True.
            verbose (bool, optional): _description_. Defaults to True.
        """

        if num_pixels is None:
            num_pixels = len(self.z), len(self.x)

        image_new, p_min, p_max = load_dxf(filename_dxf, num_pixels, filename_png, has_draw, verbose)
        # image_new = np.transpose(image_new)
        # image_new = np.flipud(image_new)
        # ok in xy but not here?

        if units == 'mm':
            p_min = p_min*1000
            p_max = p_max*1000

        if extent is None:
            self.z = np.linspace(p_min[0], p_max[0], num_pixels[0])
            self.x = np.linspace(p_min[1], p_max[1], num_pixels[1])
            self.X, self.Z = np.meshgrid(self.x, self.z)
        else:
            self.z = np.linspace(extent[0], extent[1], num_pixels[0])
            self.x = np.linspace(extent[2], extent[3], num_pixels[1])
            self.X, self.Z = np.meshgrid(self.x, self.z)

        if invert is True:
            image_new = 1-image_new

        self.n = self.n + image_new * (n_max - n_min)
        # cuidado con n_min y n_background ¿es lo mismo?

    def dots(self, positions: list[floating, floating], refractive_index: floating = 1.):
        """Generates 1 or several point masks at positions r0

        Args:
            positions (float, float) or (np.array, np.array): (x,z) point or points where mask is 1
            refractive_index (float): refractive index
        """
        x0, z0 = positions
        n = np.zeros_like(self.X)

        if type(positions[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_z0, _, _ = nearest(self.z, z0)
            n[i_x0, i_z0] = refractive_index
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_z0s, _, _ = nearest2(self.z, z0)
            for i_x0, i_z0 in zip(i_x0s, i_z0s):
                n[i_x0, i_z0] = refractive_index

        self.n = n
        return self

    def semi_plane(self, r0: list[floating, floating], refractive_index: floating | str,
                   angle: floating = 0., rotation_point: list[floating, floating] | None = None):
        """Inserts a semi-sphere in background (x>x0). If something else previous, it is removed.

        Args:
            r0=(x0,z0) (float,float): Location of the same plane.
            refractive_index (float, str): refractive index.
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0  # DUDA

        cond1 = "Zrot>{}".format(z0)

        Fs = [cond1]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def layer(self, r0: list[floating, floating], depth: floating, refractive_index: floating | str,
              angle: floating = 0., rotation_point: list[floating, floating] | None = None):
        """Insert a layer. If it is something else previous, it is removed.

        Args:
        r0 (float, float): (x0,z0) Location of the same plane, for example (0 * um, 20 * um)
        depth (float): depth of the layer
        refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def rectangle(self, r0: list[floating, floating], size: list[floating, floating],
                  refractive_index: floating | str, angle: floating = 0.,
                  rotation_point: list[floating, floating] | None = None):
        """Insert a rectangle in background. Something previous, is removed.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            size (float, float): x,z size of the rectangle
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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

        # Definition of square/rectangle
        xmin = x0 - sizex / 2
        xmax = x0 + sizex / 2
        zmin = z0 - sizez / 2
        zmax = z0 + sizez / 2

        cond1 = "Xrot<{}".format(xmax)
        cond2 = "Xrot>{}".format(xmin)
        cond3 = "Zrot<{}".format(zmax)
        cond4 = "Zrot>{}".format(zmin)

        Fs = [cond1, cond2, cond3, cond4]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={"np": np}
        )

        return ipasa

    def slit(self, r0: list[floating, floating], aperture: floating, depth: floating,
             refractive_index: floating | str, refractive_index_center: floating or str = "",
             angle: floating = 0, rotation_point: list[floating, floating] | None = None):
        """Insert a slit in background.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            aperture (float): length of the opened part of the slit
            depth (float): depth of the slit
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            refractive_index_center (float, str?): refractive index of center
                if refractive_index_center='', [], 0 then we copy what it was previously at aperture
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
            r0, refractive_index, Fs1, angle, v_globals={}
        )

        Fs2 = [cond1, cond2, cond3, cond4]
        if refractive_index_center not in ("", [], 0):
            ipasa = self.object_by_surfaces(
                r0, refractive_index_center, Fs2, angle, v_globals={}
            )
        elif refractive_index_center in ("", [], 0):
            ipasa = self.object_by_surfaces(r0, 1, Fs2, angle, v_globals={})
            self.n[ipasa] = n_back[ipasa]
        return ipasa_slit != ipasa

    def sphere(self, r0: list[floating, floating], radius: list[floating, floating],
               refractive_index: floating | str, angle: floating = 0.,
               rotation_point: list[floating, floating] | None = None):
        """Insert a sphere in background.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            radius (float, float): radius x,y of the sphere (ellipsoid)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
            x0, radiusx, z0, radiusz
        )

        Fs = [cond]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def semi_sphere(self, r0: list[floating, floating], radius: list[floating, floating],
                    refractive_index: floating | str, angle: floating = 0.,
                    rotation_point: list[floating, floating] | None = None):
        """Insert a semi_sphere in background.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            radius (float, float): radius x,y of the sphere (ellipsoid)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
            x0, radiusx, z0, radiusz
        )

        Fs = [cond1, cond2]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def lens_plane_convergent(
            self,
            r0: list[floating, floating],
            aperture: floating,
            radius: floating,
            thickness: floating,
            refractive_index: floating | str,
            angle: floating = 0.,
            rotation_point: list[floating, floating] | None = None,
            mask=0):
        """Insert a plane-convergent lens in background-

        Args:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um)
                for plane-convergent z0 is the location of the plane
                for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float): radius of the curved surface
            thickness (float): thickness at the center of the lens
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            mask (array, str):  (mask_depth, refractive_index) or False.
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
            mask_refractive_index = 1 - 0.1j
        else:
            mask_depth, mask_refractive_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_plane = "Zrot>{}".format(z_plane)
        cond_radius = "(Xrot - {})**2 +(Zrot -{})**2 <{}**2".format(
            x0, z_center_lens, radius
        )
        Fs = [cond_aperture1, cond_aperture2, cond_plane, cond_radius]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refractive_index=mask_refractive_index,
                refractive_index_center="",
                angle=angle,
                rotation_point=rotation_point,
            )
        focus = radius / (refractive_index - 1)
        return focus, ipasa

    def lens_convergent(
            self,
            r0: list[floating, floating],
            aperture: floating,
            radius: floating,
            thickness: floating,
            refractive_index: floating | str,
            angle: floating = 0.,
            rotation_point: list[floating, floating] | None = None,
            mask=0):
        """Inserts a convergent lens in background.

        Args:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float, float): (radius1,radius2) radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float): rotation point.
            mask (array, str):  (mask_depth, refractive_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

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
            mask_refractive_index = 1 - 0.1j
        else:
            mask_depth, mask_refractive_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_radius1 = "(Xrot - {})**2 +(Zrot -{})**2 <({})**2".format(
            x0, z_center_lens1, radius1
        )
        cond_radius2 = "(Xrot - {})**2 +(Zrot -{})**2 <({})**2".format(
            x0, z_center_lens2, -radius2
        )

        Fs = [cond_aperture1, cond_aperture2, cond_radius1, cond_radius2]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refractive_index=mask_refractive_index,
                refractive_index_center="",
                angle=angle,
            )

        focus_1 = (refractive_index - 1) * (
            (1 / radius1 - 1 / radius2) -
            (refractive_index - 1) *
            thickness /
            (refractive_index * radius1 * radius2)
        )
        return 1 / focus_1, ipasa

    def lens_plane_divergent(
        self,
        r0: list[floating, floating],
        aperture: floating,
        radius: floating,
        thickness: floating,
        refractive_index: floating | str,
        angle: floating = 0.,
        rotation_point: list[floating, floating] | None = None,
        mask=False,
    ):
        """Insert a plane-divergent lens in background.

        Args:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float): radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            mask (array, str):  (mask_depth, refractive_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

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
            mask_refractive_index = 1 - 0.1j
        else:
            mask_depth, mask_refractive_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_plane = "Zrot>{}".format(z0)
        cond_radius = "(Xrot - {})**2 +(Zrot -{})**2 >({})**2".format(
            x0, z_center_lens, radius
        )
        cond_right = "Zrot<{}".format(z_center_lens)
        Fs = [cond_aperture1, cond_aperture2,
              cond_plane, cond_radius, cond_right]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refractive_index=mask_refractive_index,
                refractive_index_center="",
                angle=angle,
                rotation_point=rotation_point,
            )
        focus = radius / (refractive_index - 1)
        return focus, ipasa

    def lens_divergent(
        self,
        r0: list[floating, floating],
        aperture: floating,
        radius: floating,
        thickness: floating,
        refractive_index: floating | str,
        angle: floating = 0.,
        rotation_point: list[floating, floating] | None = None,
        mask=0,
    ):
        """Insert a  divergent lens in background.

        Args:
            r0 (float, float): (x0,z0) position of the center of lens, for example (0 * um, 20 * um) for plane-convergent z0 is the location of the plane for convergent-plane (angle =180*degrees) the thickness has to be added to z0
            aperture (float): aperture of the lens. If it is 0, then it is not applied
            radius (float, float): (radius1, radius2) radius of curvature (with sign)
            thickness (float): thickness at the center of the lens
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float): rotation point
            mask (array, str):  (mask_depth, refractive_index) or False. It masks the field outer the lens using a slit with depth = mask_depth

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
            mask_refractive_index = 1 - 0.1j
        else:
            mask_depth, mask_refractive_index = mask

        if aperture > 0:
            cond_aperture1 = "Xrot<{}".format(x0 + aperture / 2)
            cond_aperture2 = "Xrot>{}".format(x0 - aperture / 2)
        else:
            cond_aperture1 = "Xrot<1e6"
            cond_aperture2 = "Xrot>-1e6"
        cond_radius1 = "(Xrot - {})**2 +(Zrot -{})**2>({})**2".format(
            x0, z_center_lens1, radius1
        )
        cond_radius2 = "(Xrot - {})**2 +(Zrot -{})**2 >({})**2".format(
            x0, z_center_lens2, -radius2
        )
        cond_right = "Zrot>{}".format(z_center_lens1)
        cond_left = "Zrot<{}".format(z_center_lens2)

        Fs = [
            cond_aperture1,
            cond_aperture2,
            cond_radius1,
            cond_radius2,
            cond_right,
            cond_left,
        ]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )

        if mask_depth > 0:
            self.slit(
                r0=r0,
                aperture=aperture,
                depth=mask_depth,
                refractive_index=mask_refractive_index,
                refractive_index_center="",
                angle=angle,
                rotation_point=rotation_point,
            )
        focus_1 = (refractive_index - 1) * (
            (1 / radius1 - 1 / radius2) -
            (refractive_index - 1) *
            thickness /
            (refractive_index * radius1 * radius2)
        )
        return 1 / focus_1, ipasa

    def aspheric_surface_z(self, r0: list[floating, floating], refractive_index: floating | str,
                           cx: floating, Qx: floating, a2: floating, a3: floating, a4: floating,
                           side: str, angle: floating = 0.):
        """Define an aspheric surface

        Args:
            r0 (float, float): (x0,z0) position of apex
            refractive_index (float, str): refractive index
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

        if side == "right":
            sign = ">"
        elif side == "left":
            sign = "<"
        else:
            print("possible error in aspheric")

        params = dict(x0=x0, z0=z0, cx=cx, Qx=Qx,
                      a2=a2, a3=a3, a4=a4, sign=sign)

        cond = "Zrot{sign}{z0}+{cx}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx})*{cx}**2*(Xrot-{x0})**2+{a2}*(Xrot-{x0})**4+{a3}*(Xrot-{x0})**6)+{a4}*(Xrot-{x0})**8)".format(
            **params
        )

        Fs = [cond]
        v_globals = {"self": self, "np": np, "degrees": degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )
        return ipasa

    def aspheric_lens(
            self,
            r0: list[floating, floating],
            angle: floating,
            refractive_index: floating | str,
            cx: list[floating, floating],
            Qx: list[floating, floating],
            depth: list[floating, floating],
            size: floating,
            a2: list[floating, floating] = (0, 0),
            a3: list[floating, floating] = (0, 0),
            a4: list[floating, floating] = (0, 0),
            a5: list[floating, floating] = (0, 0),
            a6: list[floating, floating] = (0, 0),
            a7=(0, 0)):
        """Define an aspheric surface as defined in Gomez-Pedrero.

        Args:
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
        a51, a52 = a5
        a61, a62 = a6
        a71, a72 = a7
        side1, side2 = "left", "right"

        if side1 == "right":
            sign1 = "<"
        else:
            sign1 = ">"

        if side2 == "right":
            sign2 = "<"
        else:
            sign2 = ">"

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
            a51=a51,
            a52=a52,
            a61=a61,
            a62=a62,
            a71=a71,
            a72=a72,
            d1=z0,
            d2=z0 + depth,
            sign1=sign1,
            sign2=sign2,
        )

        cond1 = "Zrot{sign1}{d1}+{cx1}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx1})*{cx1}**2*(Xrot-{x0})**2))+{a21}*(Xrot-{x0})**4+{a31}*(Xrot-{x0})**6+{a41}*(Xrot-{x0})**8+{a51}*(Xrot-{x0})**10+{a61}*(Xrot-{x0})**12+{a71}*(Xrot-{x0})**14".format(
            **params
        )

        cond2 = "Zrot{sign2}{d2}+{cx2}*(Xrot-{x0})**2/(1+np.sqrt(1-(1+{Qx2})*{cx2}**2*(Xrot-{x0})**2))+{a22}*(Xrot-{x0})**4+{a32}*(Xrot-{x0})**6+{a42}*(Xrot-{x0})**8+{a52}*(Xrot-{x0})**10+{a62}*(Xrot-{x0})**12+{a72}*(Xrot-{x0})**14".format(
            **params
        )

        cond3 = "(Xrot-{})<{}".format(x0, size / 2)
        cond4 = "(Xrot-{})>{}".format(x0, -size / 2)

        cond5 = "Zrot > {}".format(z0 - depth)
        cond6 = "Zrot < {}".format(z0 + depth)

        Fs = [cond1, cond2, cond3, cond4, cond5, cond6]
        v_globals = {"self": self, "np": np, "degrees": degrees}

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )

        return ipasa, Fs

    def wedge(
            self, r0: list[floating, floating], length, refractive_index: floating | str, angle_wedge: floating,
            angle: floating = 0., rotation_point: list[floating, floating] | None = None):
        """Insert a wedge pointing towards the light beam

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def prism(
            self, r0: list[floating, floating], length: floating, refractive_index: floating | str,
            angle_prism: floating, angle: floating = 0., rotation_point: list[floating, floating] | None = None):
        """Similar to wedge but the use is different. Also the angle is usually different. One of the sides is paralel to x=x0

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle_prism (float), angle of the prism in radians
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float). Rotation point
        """

        x0, z0 = r0
        if rotation_point is None:
            rotation_point = r0

        cond1 = "Xrot>{}".format(x0)
        cond2 = "Zrot-({})>{}*(Xrot-{})".format(z0,
                                                np.tan(angle_prism / 2), x0)
        cond3 = "Zrot-({})<{}*(Xrot-{})".format(
            z0 + length, np.tan(np.pi - angle_prism / 2), x0
        )

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def biprism(self, r0: list[floating, floating], length: floating, height: floating,
                refractive_index: floating | str, angle: floating = 0.):
        """Fresnel biprism.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the long part (z direction)
            height (float): height of biprism
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
            cond2 = "Zrot-({})<{}*(Xrot-{})".format(
                z0 + height, -2 * height / length, x0
            )
            cond3 = "Zrot-({})<{}*(Xrot-{})".format(
                z0 + height, 2 * height / length, x0
            )
        else:
            cond1 = "Zrot<{}".format(z0)
            cond2 = "Zrot-({})>{}*(Xrot-{})".format(
                z0 - height, -2 * height / length, x0
            )
            cond3 = "Zrot-({})>{}*(Xrot-{})".format(
                z0 - height, +2 * height / length, x0
            )

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals={}
        )
        return ipasa

    def ronchi_grating(
            self,
            r0: list[floating, floating],
            period: floating,
            fill_factor: floating,
            length: floating,
            height: floating,
            Dx: floating,
            refractive_index: floating | str,
            heigth_substrate: floating,
            refractive_index_substrate: floating,
            angle: floating = 0.):
        """Insert a ronchi grating in background.

        Args:
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            length (float): length of the grating
            height (float): height of the grating
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            heigth_substrate (float): height of the substrate
            refractive_index_substrate (float, str): refractive index of substrate,  1.5 + 1.0j
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
            refractive_index=refractive_index,
            angle=angle,
        )

        if heigth_substrate > 0:
            self.rectangle(
                r0, (length, heigth_substrate), refractive_index_substrate, angle
            )
        self.slit(
            r0=(x0, z0 + heigth_substrate / 2),
            aperture=length,
            depth=height,
            refractive_index=self.n_background,
            refractive_index_center="",
            angle=angle,
        )

    def sine_grating(
            self,
            period: floating,
            heigth_sine: floating,
            heigth_substrate: floating,
            r0: list[floating, floating],
            length: floating,
            Dx: floating,
            refractive_index: floating | str,
            angle: floating = 0.):
        """Insert a sine grating in background.

        Args:
            period (float): period of the grating
            fill_factor (float): [0,1], fill factor of the grating
            heigth_sine (float): height of the grating
            heigth_substrate (float): height of the substrate
            r0 (float, float): (x0,z0) Location of the rectangle, for example (0 * um, 20 * um)
            length (float): length of the grating
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
        """

        x0, z0 = r0
        Xrot, Zrot = self.__rotate__(angle, r0)

        c1 = Zrot < z0 + heigth_substrate - heigth_sine / 2 + heigth_sine / 2 * np.cos(
            2 * np.pi * (Xrot - x0 - Dx) / period
        )
        c2 = Zrot > z0
        conditionZ = c1 * c2  # no es sin, es square
        conditionX = (Xrot > x0 - length / 2) * (Xrot < x0 + length / 2)
        ipasa = conditionZ * conditionX
        self.n[ipasa] = refractive_index
        return ipasa

    def probe(self, r0: list[floating, floating], base: floating, length: floating,
              refractive_index: floating | str, angle: floating = 0.):
        """Probe with a sinusoidal shape.

        Args:
            r0 (float, float): (x0,z0) position of the center of base, for example (0 * um, 20 * um)
            base (float): base of the probe
            length (float): length of the graprobeing
            Dx (float): displacement of grating with respect x=0
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
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
        v_globals = {"self": self, "np": np, "degrees": degrees, "um": um}
        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, angle, v_globals=v_globals
        )
        return ipasa

    def rough_sheet(self, r0: list[floating, floating], size: floating, t: floating, s: floating,
                    refractive_index: floating | str, angle: floating = 0.,
                    rotation_point: list[floating, floating] | None = None):
        """Sheet with one of the surface rough.

        Args:
            r0 (float, float):(x0,z0) Location of sphere, for example (0 * um, 20 * um)
            size (float, float): (sizex, sizez) size of the sheet
            s (float): std roughness
            t (float): correlation length of roughness
            refractive_index (float, str): refractive index
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

        fx = h_corr / (k * (refractive_index - 1))  # heights

        cond1 = "Zrot>{}".format(z0)
        cond2 = "Xrot<{}".format(xmax)
        cond3 = "Xrot>{}".format(xmin)

        Fs = [cond1, cond2, cond3]

        ipasa = self.object_by_surfaces(
            rotation_point, refractive_index, Fs, 0, v_globals={}
        )

        i_z, _, _ = nearest2(self.z, z0 + sizez - fx)
        i_final = len(self.z)
        for i in range(len(self.x)):
            self.n[i, i_z[i]: i_final] = n_back[i, i_z[i]: i_final]

        if angle != 0:
            self.rotate_field(angle, rotation_point,
                              n_background=self.n_background)
        return ipasa

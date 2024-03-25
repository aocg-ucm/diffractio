# !/usr/bin/env python3

"""
This module generates Scalar_mask_XY class for definingn masks. Its parent is Scalar_field_X.

The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refractive index XZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * set_amplitude, set_phase
    * binarize, two_levels, gray_scale
    * a_dataMatrix
    * area
    * save_mask
    * inverse_amplitude, inverse_phase
    * widen
    * image
    * point_maks, slit, double_slit, square, circle, super_gauss, square_circle, ring, cross
    * mask_from_function
    * prism, lens, lens_spherical, aspheric, fresnel_lens
    * sine_grating, sine_edge_grating ronchi_grating, binary_grating, blazed_grating, forked_grating, grating2D, grating_2D_chess
    * axicon, axicon_binary, biprism_fresnel,
    * radial_grating, angular_grating, hyperbolic_grating, archimedes_spiral, laguerre_gauss_spiral
    * hammer
    * roughness, circle_rough, ring_rough, fresnel_lens_rough,
"""

import matplotlib.figure as mpfig
import matplotlib.image as mpimg
from diffractio.utils_math import cart2pol

from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import eval_hermite
import matplotlib.path as mpath

from .utils_typing import npt, Any, NDArray, floating, NDArrayFloat, NDArrayComplex


from . import degrees, np, plt, sp, um
from .scalar_fields_XY import Scalar_field_XY
from .scalar_sources_XY import Scalar_source_XY
from .utils_math import (fft_convolution2d, laguerre_polynomial_nk, nearest,
                         nearest2)
from .utils_optics import roughness_2D
from .utils_dxf import load_dxf


class Scalar_mask_XY(Scalar_field_XY):
    """Class for working with XY scalar masks.

    Args:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`
        y (numpy.array): linear array with equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 wavelength: float | None = None, info: str = ""):

        super().__init__(x, y, wavelength, info)
        self.type = 'Scalar_mask_XY'

    def set_amplitude(self, q: bool = True, positive: bool = False, amp_min: floating = 0.,
                      amp_max: floating = 1.):
        """ TODO:
        makes that the mask has only amplitude.

        Args:
            q (int): 0 - amplitude as it is and phase is removed. 1 - take phase and convert to amplitude

            positive (int): 0 - value may be positive or negative. 1 - value is only positive
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        if q == False:
            if positive is False:
                self.u = amp_min + (amp_max -
                                    amp_min) * amplitude * np.sign(phase)
            elif positive is True:
                self.u = amp_min + (amp_max - amp_min) * amplitude
        else:
            if positive is False:
                self.u = amp_min + (amp_max - amp_min) * phase
            elif positive is True:
                self.u = amp_min + (amp_max - amp_min) * np.abs(phase)

        # hay que terminar

    def set_phase(self, q: bool = True, phase_min: floating = 0., phase_max: floating = np.pi):
        """Makes the mask as phase,
            q=0: Pass amplitude to 1.
            q=1: amplitude pass to phase
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        if q == 0:
            self.u = np.exp(1.j * phase)
        if q == 1:
            self.u = np.exp(1.j * (phase_min +
                                   (phase_max - phase_min) * amplitude))

    def area(self, percentage: floating):
        """Computes area where mask is not 0

        Args:
            percentage_maximum (float): percentage from maximum intensity to compute

        Returns:
            float: area (in um**2)

        Example:
            area(percentage=0.001)
        """

        intensity = np.abs(self.u)**2
        max_intensity = intensity.max()
        num_pixels_1 = sum(sum(intensity > max_intensity * percentage))
        num_pixels = len(self.x) * len(self.y)
        delta_x = self.x[1] - self.x[0]
        delta_y = self.y[1] - self.y[0]

        return (num_pixels_1 / num_pixels) * (delta_x * delta_y)

    def inverse_amplitude(self, new_field: bool = False):
        """Inverts the amplitude of the mask, phase is equal as initial

        Args:
            new_field (bool): If True it returns a Scalar_mask_XY object, else, it modifies the existing object


        Returns:
            Scalar_mask_XY:  If new_field is True, it returns a Scalar_mask_XY object.
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        new_amplitude = (1 - amplitude) * np.exp(1.j * phase)

        if new_field is False:
            self.u = new_amplitude
        else:
            new = Scalar_mask_XY(self.x, self.y, self.wavelength)
            new.u = new_amplitude
            return new

    def inverse_phase(self, new_field: bool = False):
        """Inverts the phase of the mask, amplitude is equal as initial

        Args:
            new_field (bool): If True it returns a Scalar_mask_XY object, else, it modifies the existing object


        Returns:
            Scalar_mask_XY:  If new_field is True, it returns a Scalar_mask_XY object.
        """

        amplitude = np.abs(self.u)
        phase = np.angle(self.u)

        new_amplitude = amplitude * np.exp(-1.j * phase)

        if new_field is False:
            self.u = new_amplitude
        else:
            new = Scalar_mask_XY(self.x, self.y, self.wavelength)
            new.u = new_amplitude
            return new

    def filter(self, mask, new_field: bool = True, binarize=False, normalize: bool = False):
        """Widens a field using a mask

        Args:
            mask (diffractio.Scalar_mask_XY): filter
            new_field (bool): If True, develope new Field
            binarize (bool, float): If False nothing, else binarize in level
            normalize (bool): If True divides the mask by sum.
        """

        f1 = np.abs(mask.u)

        if normalize is True:
            f1 = f1 / f1.sum()

        covolved_image = fft_convolution2d(f1, np.abs(self.u))
        if binarize is not False:
            covolved_image[covolved_image > binarize] = 1
            covolved_image[covolved_image <= binarize] = 0

        if new_field is True:
            new = Scalar_field_XY(self.x, self.y, self.wavelength)
            new.u = covolved_image
            return new
        else:
            self.u = covolved_image

    def widen(self, radius: float, new_field: bool = True, binarize=True):
        """Widens a mask using a convolution of a certain radius

        Args:
            radius (float): radius of convolution
            new_field (bool): returns a new XY field
            binarize (bool): binarizes result.
        """

        filter = Scalar_mask_XY(self.x, self.y, self.wavelength)
        filter.circle(r0=(0 * um, 0 * um), radius=radius, angle=0 * degrees)

        image = np.abs(self.u)
        filtrado = np.abs(filter.u) / np.abs(filter.u.sum())

        covolved_image = fft_convolution2d(image, filtrado)
        minimum = 0.01 * covolved_image.max()

        if binarize is True:
            covolved_image[covolved_image > minimum] = 1
            covolved_image[covolved_image <= minimum] = 0
        else:
            covolved_image = covolved_image / covolved_image.max()

        if new_field is True:
            filter.u = covolved_image
            return filter
        else:
            self.u = covolved_image

    # __MASKS____________________________________________

    def extrude_mask_x(self, mask_X, y0: float = None, y1: float = None, kind: str = 'unique',
                       normalize: bool = None):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Args:
            mask_X (Scalar_mask_X): an amplitude mask of type Scalar_mask_X.
            y0 (float): initial  position of mask
            y1 (float): final position of mask
            kind (str): 'superpose', 'unique'
            normalize (str): if 'cut' (>1 -> 1), 'normalize', None
        """

        if y0 is None:
            y0 = self.y[0]
        if y1 is None:
            y1 = self.y[-1]

        iy0, _, _ = nearest(vector=self.y, number=y0)
        iy1, _, _ = nearest(vector=self.y, number=y1)

        for i, index in enumerate(range(iy0, iy1)):
            if kind == 'unique':
                self.u[index, :] = mask_X.u
            elif kind == 'superpose':
                self.u[index, :] = self.u[index, :] + mask_X.u

        if normalize == 'cut':
            self.u[self.u > 1] = 1
        elif normalize == 'normalize':
            maximum = np.abs(self.u.max())
            self.u = self.u / maximum

    def mask_from_function(self, r0: floating | list, index: floating, f1, f2, radius: floating = 0,
                           v_globals: dict = {}):
        """ phase mask defined between 2 surfaces $f_1$ and $f_2$:  $h(x,y)=f_2(x,y)-f_1(x,y)$

        Args:
            r0 (float, float): center of cross
            index (float): refractive index
            f1 (str): function for first surface
            f2 (str): function for second surface
            radius (float, float) or (float): size of mask
            v_globals (dict): dictionary with globals
            mask (bool): If True applies mask
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        k = 2 * np.pi / self.wavelength

        if radius[0] > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, 0 * degrees)
            t = amplitude.u
        else:
            t = 1

        v_locals = {'self': self, 'sp': sp, 'degrees': degrees}

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)
        self.u = t * np.exp(1.j * k * (index - 1) * (F2 - F1))
        self.u[t == 0] = 0

    def image(self,
              filename: str = '',
              channel: int = 0,
              normalize: bool = True,
              lengthImage: bool = False,
              invert: bool = False,
              angle: floating = 0):
        """Converts an image file XY mask. If the image is color, we get the first Red frame

        Args:
            filename (str): filename of the image
            channel (int): number of channel RGB to get the image
            normalize (bool): if True normalizes the image
            lengthImage (bool, int): If False does nothing, if number resize image
            invert (bool): if True the image is inverted
            angle (float): rotates image a certain angle

        Returns
            str: filename
    """

        # Abre image (no la muestra)
        im = Image.open(filename)

        # Image traspuesta
        im = im.transpose(1)
        # Extrae sus components de color en varios canales
        colores = im.split()

        # Seleccionamos un channel de color
        image = colores[channel]

        # data = image.getdata()

        # Reajuste del length manteniendo la relacion de aspecto
        if lengthImage is False:
            length = self.u.shape
            image = image.resize(length)

        if lengthImage is True:
            length = im.size
            self.x = np.linspace(self.x[0], self.x[-1], length[0])
            self.y = np.linspace(self.y[0], self.y[-1], length[1])
            self.X, self.Y = np.meshgrid(self.x, self.y)

        # Rotacion de la image
        if angle != 0:
            image = image.rotate(angle)

        data = np.array(image)
        # Inversion de color
        if invert is True:
            data = data.max() - data

        # Normalizacion de la intensity
        if normalize is True:
            data = (data - data.min()) / (data.max() - data.min())

        self.u = data
        return filename

    def dxf(self, filename_dxf: str, num_pixels: list[int, int] | None = None,
            extent: list[float] | None = None, units: str = 'um', invert: bool = False,
            verbose: bool = False):
        """Loads a dxf file. Internally it has the extension of the drawing, so it is not required to generate x,y spaces. It is possible with extent, but then the file is scaled. Warning: Dxf files are usually in mm. and diffractio works in um. To generate .u, a temporal .png file is generated. 
        If x and y arrays are given, then num_pixels and extent are not used.

        msp.units = 13 # 0 - sin ,  4 mm,   12 nm,  13 um,


        Args:
            filename_dxf (str): DXF filename .dxf
            num_pixels (list[int, int] | None, optional): If . Defaults to None.
            extent (_type_, optional): _description_. Defaults to None.
            units (str, optional): _description_. Defaults to 'mm'.
            invert (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
        """

        if self.x is not None:
            num_pixels = len(self.x), len(self.y)

        image_new, p_min, p_max, msp = load_dxf(filename_dxf, num_pixels, verbose)
        image_new = np.flipud(image_new)

        if units == 'mm':
            p_min = p_min*1000
            p_max = p_max*1000
        elif units == 'inches':
            p_min = p_min*25400
            p_max = p_max*25400

        if self.x is None:

            if extent is None:

                self.x = np.linspace(p_min[0], p_max[0], num_pixels[0])
                self.y = np.linspace(p_min[1], p_max[1], num_pixels[1])
                self.X, self.Y = np.meshgrid(self.x, self.y)
            else:
                self.x = np.linspace(extent[0], extent[1], num_pixels[0])
                self.y = np.linspace(extent[2], extent[3], num_pixels[1])
                self.X, self.Y = np.meshgrid(self.x, self.y)

        if invert is True:
            image_new = 1-image_new

        self.u = image_new

    def repeat_structure(self,
                         num_repetitions: int,
                         position: str = 'center',
                         new_field: bool = True):
        """Repeat the structure (n x m) times.

        Args:
            num_repetitions (int, int): Number of repetitions of the mask
            position (string or number,number): 'center', 'previous' or initial position. Initial x
            new_field (bool): If True, a new mask is produced, else, the mask is modified.

        """

        u0 = self.u
        x0 = self.x
        y0 = self.y
        wavelength = self.wavelength

        u_new = np.tile(u0, (num_repetitions[1], num_repetitions[0]))

        x_min = x0[0]
        x_max = x0[-1]
        # dx = x0[1] - x0[0]

        y_min = y0[0]
        y_max = y0[-1]
        # dy = y0[1] - y0[0]

        x_new = np.linspace(num_repetitions[0] * x_min,
                            num_repetitions[0] * x_max,
                            num_repetitions[0] * len(x0))
        y_new = np.linspace(num_repetitions[1] * y_min,
                            num_repetitions[1] * y_max,
                            num_repetitions[1] * len(y0))

        center_x = (x_new[-1] + x_new[0]) / 2
        center_y = (y_new[-1] + y_new[0]) / 2

        if position == 'center':
            x_new = x_new - center_x
            y_new = y_new - center_y

        elif position == 'previous':
            x_new = x_new - x_new[0] + x0[0]
            y_new = y_new - y_new[0] + y0[0]

        elif isinstance(position, np.array):
            x_new = x_new - x_new[0] + position[0]
            y_new = y_new - y_new[0] + position[1]

        if new_field is True:
            t_new = Scalar_mask_XY(x=x_new, y=y_new, wavelength=wavelength)
            t_new.u = u_new

            return t_new

        else:
            self.u = u_new
            self.x = x_new
            self.y = y_new

    def masks_to_positions(self, pos: list[floating] | NDArrayFloat, new_field: bool = True,
                           binarize: bool = False, normalize: bool = False):
        """
        Place a certain mask on several positions.

        Args:
        pos (float, float) or (np.array, np.array): (x,y) point or points where mask is placed.
        new_field (bool): If True, a new mask is produced, else, the mask is modified. Default: True.
        binarize (bool, float): If False nothing, else binarize in level. Default: False.
        normalize (bool): If True divides the mask by sum. Default: False.

        Example:
            masks_to_positions(np.array([[0,100,100],[0,-100,100]]),new_field=True)
        """

        lens_array = Scalar_mask_XY(self.x, self.y, self.wavelength)
        lens_array.dots(r0=pos)

        f1 = self.u

        if normalize is True:
            f1 = f1 / f1.sum()

        covolved_image = fft_convolution2d(f1, lens_array.u)

        if binarize is not False:
            covolved_image[covolved_image > binarize] = 1
            covolved_image[covolved_image <= binarize] = 0

        if new_field is True:
            new = Scalar_field_XY(self.x, self.y, self.wavelength)
            new.u = covolved_image
            return new
        else:
            self.u = covolved_image

    def polygon(self, vertices: NDArrayFloat):
        """Draws a polygon with the vertices given in a Nx2 numpy array.

        Args:
            vertices (np.array): Nx2 array with the x,y positions of the vertices.

        Example:
            x0 = np.linspace(-3 * mm, 3 * mm, 512)
            y0 = np.linspace(-3 * mm, 3 * mm, 512)
            wavelength = 0.6328 *um
            vertices = np.array([(0 * mm, 0 * mm), (2 * mm, 0 * mm), (2 * mm, 1 * mm),(0 * mm, 1 * mm)])
            t = Scalar_mask_XY(x0, y0, wavelength)
            t.polygon(vertices)
            t.draw()
        """

        num_x, num_y = self.u.shape

        verticesx, _, _ = nearest2(self.y, vertices[:, 1])
        verticesy, _, _ = nearest2(self.x, vertices[:, 0])

        i_vertices = np.column_stack((verticesx, verticesy))

        # Create the coordinates of the matrix
        coordinates = np.column_stack(
            (np.repeat(np.arange(num_x),
                       num_y), np.tile(np.arange(num_y), num_x)))

        # Create the Path object of the polygon
        path = mpath.Path(i_vertices)

        # Verificar si cada punto de la matriz está dentro del polígono
        in_polygon = path.contains_points(coordinates)

        # Check if each point in the array is inside the polygon
        self.u[in_polygon.reshape((num_x, num_y))] = 1

    def regular_polygon(self, num_vertices: int, radius: floating, angle: floating = 0.):
        """Generates a regular polygon.

        Args:
            num_vertices (int): num_vertices
            radius (float): external radius
            angle (float): angle of rotation

        Returns:
            vertices (np.array): position of vertices

        """

        i_vertices = np.array(range(num_vertices + 1))
        angles = 2 * np.pi * i_vertices / num_vertices

        x_vertices = radius * np.cos(angles - angle + 90 * degrees)
        y_vertices = radius * np.sin(angles - angle + 90 * degrees)

        vertices = np.column_stack((x_vertices, y_vertices))

        self.polygon(vertices)

        return vertices

    def star(self, num_peaks: int, radii: list[floating], angle: floating = 0.):
        """Generates a regular polygon

        Args:
            num_peaks (int): number of peaks.
            radii (float, float): external radius
            angle (float): angle of rotation

        Returns:
            vertices (np.array): position of vertices

        Example:
            x0 = np.linspace(-3 * mm, 3 * mm, 512)
            y0 = np.linspace(-3 * mm, 3 * mm, 512)
            wavelength = 0.6328 *um
            t = Scalar_mask_XY(x0, y0, wavelength)
            vertices = t.stars(5, (2*mm,1*mm), 0*degrees)
            t.draw()

        """
        radii = np.array(radii)

        i_vertices = np.array(range(num_peaks))
        angles = 2 * np.pi * i_vertices / num_peaks

        phase_shift = 2 * np.pi / (2 * num_peaks)

        x_vertices_max = radii[0] * np.cos(angles - angle + 90 * degrees)
        y_vertices_max = radii[0] * np.sin(angles - angle + 90 * degrees)
        vertices_max = np.column_stack((x_vertices_max, y_vertices_max))

        x_vertices_min = radii[1] * np.cos(angles - angle + phase_shift +
                                           90 * degrees)
        y_vertices_min = radii[1] * np.sin(angles - angle + phase_shift +
                                           90 * degrees)
        vertices_min = np.column_stack((x_vertices_min, y_vertices_min))

        # Find the maximum number of rows between both matrices
        max_num_rows = 2 * num_peaks

        # Create an empty interleaved matrix with twice the number of rows
        interleaved_matrix = np.empty((max_num_rows, 2))

        # Interleave the rows from both matrices
        interleaved_matrix[:max_num_rows:2] = vertices_max
        interleaved_matrix[1:max_num_rows:2] = vertices_min

        self.polygon(interleaved_matrix)

        return interleaved_matrix

    def triangle(self, r0: list[floating], slope: floating, height: floating, angle: floating):
        """Create a triangle mask. It uses the equation of a straight line: y = -slope * (x - x0) + y0

        Args:
            r0 (float, float): Coordinates of the top corner of the triangle
            slope (float): Slope if the equation above
            height (float): Distance between the top corner of the triangle and the basis of the triangle
            angle (float): Angle of rotation of the triangle
        """
        if isinstance(r0, (float, int)):
            x0, y0 = (r0, r0)
        elif r0 is None:
            x0 = 0 * um
            y0 = height / 2
        else:
            x0, y0 = r0

        # Rotation of the super-ellipse
        Xrot, Yrot = self.__rotate__(angle)

        Y = -slope * np.abs(Xrot - x0) + y0
        u = np.zeros_like(self.X)

        ipasa = (Yrot < Y) & (Yrot > y0 - height)
        u[ipasa] = 1
        u[u > 1] = 1
        self.u = u

    def photon_sieve(self, t1, r0: list[floating], top_one: bool = True):
        """Generates a matrix of shapes given in t1.

        Args:
            t1 (Scalar_mask_XY): Mask of the desired figure to be drawn
            r0 (float, float) or (np.array, np.array): (x,y) point or points where mask is 1
            top_one (bool): If True, max(mask) = 1

        Returns:
            (int): number of points in the mask
        """
        r0 = np.array(r0)
        x0 = r0[:, 0]
        y0 = r0[:, 1]
        u = np.zeros_like(self.X)
        uj = np.zeros_like(self.X)

        if type(r0[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_y0, _, _ = nearest(self.y, y0)
            u[i_x0, i_y0] = 1
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_y0s, _, _ = nearest2(self.y, y0)

        for i, x_i in enumerate(x0):
            y_j = y0[i]
            i_xcercano, _, _ = nearest(self.x, x_i)
            j_ycercano, _, _ = nearest(self.y, y_j)
            if x_i < self.x.max() and x_i > self.x.min() and y_j < self.y.max(
            ) and y_j > self.y.min():
                uj[i_xcercano, j_ycercano] = 1
        num_points = int(uj.sum())
        u = fftconvolve(uj, t1.u, mode='same')
        if top_one:
            A = np.abs(u)
            phase = np.angle(u)
            A[A > 1] = 1
            u = A * np.exp(1j * phase)
            # u[u > 1] = 1
        self.u = u
        return num_points

    def insert_array_masks(self, t1, space: list[floating], margin: list[floating] | floating = 0,
                           angle: floating = 0 * degrees):
        """Generates a matrix of shapes given in t1.

        Args:
            t1 (Scalar_mask_XY): Mask of the desired figure to be drawn
            space (float, float) or (float): spaces between figures.
            margin (float, float) or (float): extra space outside the mask
            angle (float): Angle to rotate the matrix of circles

        Returns:
            (int): number of points in the mask

        Example:

            A = Scalar_mask_XY(x, y, wavelength)

            A.ring(r0, radius1, radius2, angle)

            insert_array_masks(t1 = A, space = 50 * um, angle = 0 * degrees)
        """

        if isinstance(space, (int, float)):
            delta_x, delta_y = (space, space)
        else:
            delta_x, delta_y = space

        if isinstance(margin, (float, int)):
            margin_x, margin_y = (margin, margin)
        else:
            margin_x, margin_y = margin

        assert delta_x > 0 and delta_y > 0

        uj = np.zeros_like(self.X)

        X = margin_x + np.arange(self.x.min(), self.x.max() + delta_x, delta_x)
        Y = margin_y + np.arange(self.y.min(), self.y.max() + delta_y, delta_y)
        for i, x_i in enumerate(X):
            i_xcercano, _, _ = nearest(self.x, x_i)
            for j, y_j in enumerate(Y):
                j_ycercano, _, _ = nearest(self.y, y_j)
                if x_i < self.x.max() and x_i > self.x.min(
                ) and y_j < self.y.max() and y_j > self.y.min():
                    uj[i_xcercano, j_ycercano] = 1
        num_points = int(uj.sum())
        u = fftconvolve(uj, t1.u, mode='same')
        u[u > 1] = 1
        self.u = u
        return num_points

    def dots(self, r0: list[floating]):
        """Generates 1 or several point masks at positions r0

        Args:
            r0 (float, float) or (np.array, np.array): (x,y) point or points where mask is 1

        """
        x0, y0 = r0
        u = np.zeros_like(self.X)

        if type(r0[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_y0, _, _ = nearest(self.y, y0)
            u[i_y0, i_x0] = 1
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_y0s, _, _ = nearest2(self.y, y0)
            for (i_x0, i_y0) in zip(i_x0s, i_y0s):
                u[i_y0, i_x0] = 1

        self.u = u

    def dots_regular(self, xlim: list[floating], ylim: list[floating], num_data: int,
                     verbose: bool = False):
        """Generates n x m or several point masks.

        Args:
            xlim (float, float): (xmin, xmax) positions
            ylim (float, float): (ymin, ymax) positions
            num_data (int, int): (x, y) number of points
        """
        x0, x1 = xlim
        y0, y1 = ylim
        nx, ny = num_data
        x_points = np.linspace(x0, x1, nx)
        y_points = np.linspace(y0, y1, ny)

        u = np.zeros_like(self.X)
        i_x0, _, _ = nearest2(self.x, x_points)
        i_y0, _, _ = nearest2(self.y, y_points)
        if verbose is True:
            print(i_x0)
            print(i_y0)

        iX, iY = np.meshgrid(i_x0, i_y0)
        u[iX, iY] = 1

        self.u = u

    def one_level(self, level: floating = 0):
        """Sets one level for all the image.

        Args:
            level (float): value
        """
        self.u = level * np.ones(self.X.shape)

    def two_levels(self, level1: floating = 0, level2: floating = 1,
                   x_edge: floating = 0., angle: floating = 0.):
        """Divides the field in two levels

        Args:
            level1 (float): value of first level
            level2 (float): value of second level
            x_edge (float): position of division
            angle (float): angle of rotation in radians
        """
        Xrot, Yrot = self.__rotate__(angle, (x_edge, 0))
        self.u = level1 * np.ones(self.X.shape)
        self.u[Xrot > 0] = level2

    def edge_series(self,
                    r0: list[floating],
                    period: floating,
                    a_coef: NDArrayFloat,
                    b_coef: NDArrayFloat | None = None,
                    angle: floating = 0 * degrees,
                    invert: bool = True):
        """Creates a linear aperture using the Fourier coefficients.

            Args:
                x0 (float): x-axis displacement (for 'fslit' function)
                period (float): Function period

                a_coef (np.array, 2 rows and x columns): coefficients that multiply the cosine function.
                b_coef (np.array, 2 rows and x columns): coefficients that multiply the sine function.
                angle (float): angle of rotation in radians
                invert (bool): inverts transmittance values (for 'fslit' function)

                For both arrays:
                First row: coefficient orders
                Second row: coefficient values

            Example:
                t1.edge_series(x0=0, period=50, a_coef=np.array(
                    [[0,1],[100,50]]), angle = 0 * degrees, invert=False)
            """

        Xrot, Yrot = self.__rotate__(angle)
        Yrot = Yrot

        x0, y0 = r0

        # Definicion de la transmitancia
        u = np.zeros_like(self.X)

        asol = a_coef[1][0] / 2
        bsol = 0

        _, num_coefs_a = a_coef.shape
        for i in range(num_coefs_a):
            asol = asol + \
                a_coef[1][i] * np.cos(2 * np.pi * a_coef[0]
                                      [i] * (Yrot - y0) / period)

        if b_coef is not None:
            _, num_coefs_b = b_coef.shape
            for i in range(num_coefs_b):
                bsol = bsol + \
                    b_coef[1][i] * np.sin(2 * np.pi *
                                          b_coef[0][i] * (Yrot - y0) / period)

        sol = asol + bsol

        if invert is True:
            u[(Xrot - x0 > sol)] = 1
            u[(Xrot - x0 < sol)] = 0
        else:
            u[(Xrot - x0 < sol)] = 1
            u[(Xrot - x0 > sol)] = 0

        self.u = u

    def slit(self, x0: floating, size: floating, angle: floating = 0.):
        """Slit: 1 inside, 0 outside

        Args:
            x0 (float): center of slit
            size (float): size of slit
            angle (float): angle of rotation in radians
        """
        # Definicion de la slit
        xmin = -size / 2
        xmax = +size / 2

        # Rotacion de la slit
        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        # Definicion de la transmitancia
        u = np.zeros(np.shape(self.X))
        ix = (Xrot < xmax) & (Xrot > xmin)
        u[ix] = 1
        self.u = u

    def slit_series(self,
                    x0: floating,
                    width: floating,
                    period1: floating,
                    period2: floating,
                    Dy: floating,
                    a_coef1: NDArrayFloat,
                    a_coef2: NDArrayFloat,
                    b_coef1: NDArrayFloat | None = None,
                    b_coef2: NDArrayFloat | None = None,
                    angle: floating = 0.):
        """Creates a lineal function using the Fourier coefficients.

            Args:
                x0 (float): position of the center of the slit
                width (float): slit width
                period1 (float): Period of the first function
                period2 (float): Period of the second function
                Dy (float, float): Shifts of the edges
                a_coef1 (np.array, 2 rows and x columns): coefficients that multiply the cosine in the first function.
                a_coef2 (np.array, 2 rows and x columns): coefficients that multiply the cosine in the second function.
                b_coef1 (np.array, 2 rows and x columns): coefficients that multiply the sine in the first function.
                b_coef2 (np.array, 2 rows and x columns): coefficients that multiply the sine in the second function.
                For the arrays: First row - coefficient orders, Second row - coefficient values
                angle (float): angle of rotation in radians

            Example:
                t1.slit_series(x0=0, width=10, period1=50,
                               period2=20, a_coef1=np.array([[0,1],[100,50]]) )
            """
        dy1, dy2 = Dy

        t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        t1.edge_series(r0=(x0 - width / 2, dy1),
                       period=period1,
                       a_coef=a_coef1,
                       b_coef=b_coef1,
                       angle=angle,
                       invert=True)
        t2 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        t2.edge_series(r0=(x0 + width / 2, dy2),
                       period=period2,
                       a_coef=a_coef2,
                       b_coef=b_coef2,
                       angle=angle,
                       invert=False)

        self.u = t1.u * t2.u

    def double_slit(self, x0: floating, size: floating, separation: floating,
                    angle: floating = 0 * degrees):
        """double slit: 1 inside, 0 outside

        Args:
            x0 (float): center of double slit
            size (float): size of slit
            separation (float): separation between slit centers
            angle (float): angle of rotation in radians
        """

        slit1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        slit2 = Scalar_mask_XY(self.x, self.y, self.wavelength)

        # Definicion de las dos slits
        slit1.slit(x0=x0 - separation / 2, size=size, angle=angle)
        slit2.slit(x0=x0 + separation / 2, size=size, angle=angle)

        self.u = slit1.u + slit2.u

    def square(self, r0: list[floating], size: floating, angle: floating = 0.):
        """Square: 1 inside, 0 outside

        Args:
            r0 (float, float): center of square
            size (float, float) or (float): size of slit
            angle (float): angle of rotation in radians

        Example:

            m.square(r0=(0 * um, 0 * um), size=(250 * \
                     um, 120 * um), angle=0 * degrees)
        """

        # si solamente un numero, posiciones y radius son los mismos para ambos

        if isinstance(size, (float, int)):
            sizex, sizey = size, size
        else:
            sizex, sizey = size

        x0, y0 = r0

        # Definicion del square/rectangle
        xmin = -sizex / 2
        xmax = +sizex / 2
        ymin = -sizey / 2
        ymax = +sizey / 2

        # Rotacion del square/rectangle
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Transmitancia de los points interiores
        u = np.zeros(np.shape(self.X))
        ipasa = (Xrot < xmax) & (Xrot > xmin) & (Yrot < ymax) & (Yrot > ymin)
        u[ipasa] = 1
        self.u = u

    def gray_scale(self, num_levels: int, level_min: floating = 0., level_max: floating = 1.):
        """Generates a number of strips with different amplitude

        Args:
            num_levels (int): number of levels
            level_min (float): value of minimum level
            level_max (float): value of maximum level
        """
        t = np.zeros(self.X.shape, dtype=float)

        xpos = np.linspace(self.x[0], self.x[-1], num_levels + 1)
        height_levels = np.linspace(level_min, level_max, num_levels)
        ipos, _, _ = nearest2(self.x, xpos)
        ipos[-1] = len(self.x)

        for i in range(num_levels):
            t[:, ipos[i]:ipos[i + 1]] = height_levels[i]

        self.u = t

    def circle(self, r0: list[floating], radius: floating, angle: floating = 0.):
        """Creates a circle or an ellipse.

        Args:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            circle(r0=(0 * um, 0 * um), radius=(250 * um, 125 * um), angle=0 * degrees)
        """
        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        u = np.zeros(np.shape(self.X))
        ipasa = Xrot**2 / radiusx**2 + Yrot**2 / radiusy**2 < 1
        u[ipasa] = 1
        self.u = u

    def circular_sector(self, r0: list[floating], radii: floating | list[floating], angles: floating):
        """Generates a circular sector.

        Args:
            r0 (int, int): position of center
            radii (float) or (float, float): radius
            angles (float, float): initial and final angle in radians.

        """

        if isinstance(radii, float):
            radii = (0, radii)

        [rho, theta] = cart2pol(self.X - r0[0], self.Y - r0[1])

        ix = (theta > angles[0]) & (theta <= angles[1]) & (rho >= radii[0]) & (
            rho < radii[1])
        self.u[ix] = 1

    def super_gauss(self, r0: list[floating], radius: list[floating] | floating,
                    power: floating = 2, angle: floating = 0.):
        """Supergauss mask.

        Args:
            r0 (float, float): center of circle
            radius (float, float) or (float): radius of circle
            power (float): value of exponential
            angle (float): angle of rotation in radians

        Example:

            super_gauss(r0=(0 * um, 0 * um), radius=(250 * um,
                        125 * um), angle=0 * degrees, potencia=2)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Radios mayor y menor
        x0, y0 = r0

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))
        R = np.sqrt(Xrot**2 + Yrot**2)
        self.u = np.exp(-R**power / (2 * radiusx**power))

    def square_circle(self, r0: floating, R1: floating, R2: floating, s: floating, angle: floating = 0.):
        """ Between circle and square, depending on fill factor s

        s=0 circle, s=1 square

        Args:
            r0 (float, float): center of square_circle
            R1 (float): radius of first axis
            R2 (float): radius of first axis
            s (float): [0-1] shape parameter: s=0 circle, s=1 square
            angle (float): angle of rotation in radians

        Reference:
            M. Fernandez Guasti, M. De la Cruz Heredia "diffraction pattern of a circle/square aperture" J.Mod.Opt. 40(6) 1073-1080 (1993)

        """
        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t1.square(r0=r0, size=(2 * R1, 2 * R2), angle=angle)
        x0, y0 = r0

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))
        F = np.sqrt(Xrot**2 / R1**2 + Yrot**2 / R2**2 - s**2 * Xrot**2 * Yrot**2 /
                    (R1**2 * R2**2))

        Z1 = F < 1
        Z = Z1 * t1.u

        self.u = Z

    def angular_aperture(self, a_coef: NDArrayFloat, b_coef: NDArrayFloat | None = None, angle: floating = 0.):
        """Creates a radial function using the Fourier coefficients.

            Args:

                a_coef (np.array, 2 rows and x columns): coefficients that multiply the cosine function.
                b_coef (np.array, 2 rows and x columns): coefficients that multiply the sine function.
                angle (float): angle of rotation in radians

                For a_coef and b_coef, the first row are the coefficient orders  and the second row are coefficient values.
            Example:

                angular_aperture(t, a_coef=np.array(
                    [[0,1],[20,10]]),  angle= 0 * degrees)
            """

        Xrot, Yrot = self.__rotate__(angle)

        # Definicion de la transmitancia
        u = np.zeros_like(self.X)

        r = np.sqrt(Xrot**2 + Yrot**2)

        phi = np.arctan2(Yrot, Xrot)

        asol = 0
        bsol = 0

        _, num_coefs_a = a_coef.shape
        for i in range(num_coefs_a):
            asol = asol + a_coef[1][i] * np.cos(a_coef[0][i] * phi)

        if b_coef is not None:
            _, num_coefs_b = b_coef.shape
            for i in range(num_coefs_b):
                bsol = bsol + b_coef[1][i] * np.sin(b_coef[0][i] * phi)

        sol = asol + bsol

        ipasa = r - abs(sol) < 0
        u[ipasa] = 1
        self.u = u
        return ipasa

    def ring(self, r0: list[floating], radius1: list[floating], radius2: list[floating], angle: list[floating] = 0.):
        """ Ring.

        Args:
            r0 (float, float): center of ring
            radius1 (float, float) or (float): inner radius
            radius2 (float, float) or (float): outer radius
            angle (float): angle of rotation in radians
        """

        ring1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring1.circle(r0, radius1, angle)
        ring2.circle(r0, radius2, angle)

        self.u = np.abs(ring2.u - ring1.u)

    def rings(self, r0: list[floating], inner_radius: NDArrayFloat, outer_radius: NDArrayFloat):
        """Structure based on several rings, with radius given by inner_radius and outer_radius.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            inner_radius (np.array): inner radius
            outer_radius (np.array): inner radius
            mask (bool): if True, mask with size radius of maximum outer radius
        """

        x0, y0 = r0
        angle = 0

        radius = outer_radius.max()

        u = np.zeros_like(self.X)
        ring = Scalar_mask_XY(self.x, self.y, self.wavelength)

        num_rings = len(inner_radius)

        for i in range(num_rings):
            ring.ring(r0, inner_radius[i], outer_radius[i], angle)
            u = u + ring.u

        self.u = u

    def cross(self, r0: list[floating], size: floating | list[floating], angle: floating = 0.):
        """ Cross

        Args:
            r0 (float, float): center of cross
            size (float, float) or (float): length, width of cross
            angle (float): angle of rotation in radians
        """
        # Definicion del origen y length de la cross

        # if isinstance(size, (float, int)):
        #     sizex, sizey = size, size
        # else:
        #     sizex, sizey = size

        # Definicion de la cross
        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        # Se define una primera mask cuadrada
        t1.square(r0, size, angle)
        # Una segunda mask cuadrada rotada 90º respecto de la anterior
        t2.square(r0, size, angle + 90 * degrees)
        # La superposicion de ambas da lugar a la cross
        t3 = t1.u + t2.u
        t3[t3 > 0] = 1

        self.u = t3

    def prism(self, r0: list[floating], angle_wedge: floating, angle: floating = 0.):
        """prism which produces a certain angle

        Args:
            r0 (float, float): center wedge
            angle_wedge (float): angle of wedge in x direction
            angle (float): angle of rotation in radians

        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        self.u = np.exp(1j * k * (Xrot) * np.sin(angle_wedge))

    def lens(self, r0: list[floating], focal: float | NDArrayFloat, radius: float = 0,
             angle: float = 0.):
        """Transparent lens

        Args:
            r0 (float, float): (x0,y0) - center of lens
            focal (float, float) or (float): focal length of lens
            radius (float, float) or (float): radius of lens mask.
                If radius = 0, no mask is applied
            angle (float): angle of axis in radians

        Example:
            lens(r0=(0 * um, 0 * um),focal=(5 * mm, 10 * mm), radius=100*um, angle: 0.)
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)
        if isinstance(focal, (float, int, complex)):
            focal = (focal, focal)

        k = 2 * np.pi / self.wavelength

        x0, y0 = r0
        f1, f2 = focal

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        if radius[0] > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = 1

        self.u = t * np.exp(-1.j * (k * ((Xrot**2 / (2 * f1)) + Yrot**2 /
                                         (2 * f2)) + np.pi))

    def lens_spherical(self, r0: list[floating], focal: floating, refractive_index: floating = 1.5,
                       radius: floating = 0.):
        """Spherical lens, without paraxial approximation. The focal distance and the refractive index are used for the definition.
        When the refractive index decreases, the radius of curvature decrases and less paraxial.
        Now, only one focal.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            focal (float): focal length of lens
            refractive_index (float): refractive index
            radius (float): radius of lens mask

        lens_spherical:
            lens(r0=(0 * um, 0 * um), radius= 200 * um, focal= 10 * mm, refractive_index=1.5)
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        k = 2 * np.pi / self.wavelength

        x0, y0 = r0
        angle = 0.

        R = (refractive_index - 1) * focal

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        if radius[0] > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = 1

        h = (np.sqrt(R**2 - (Xrot**2 + Yrot**2)) - R)

        h[R**2 - (Xrot**2 + Yrot**2) < 0] = 0
        self.u = t * np.exp(1j * k * (refractive_index - 1) * h)
        self.u[t == 0] = 0

        return h

    def aspheric(self, r0: list[floating], c: floating, k: floating, a: list, n0: floating, n1: floating,
                 radius: floating = 0.):
        """asferic surface.

        $z = \frac{c r^2}{1+\np.sqrt{1-(1+k) c^2 r^2 }}+\sum{a_i r^{2i}}$

        Args:
            x0 (float): position of center
            c (float): base curvature at vertex, inverse of radius
            k (float): conic constant
            a (list): order aspheric coefficients: A4, A6, A8, ...
            n0 (float): refractive index of first medium
            n1 (float): refractive index of second medium
            radius (float): radius of aspheric surface

            Conic Constant    Surface Type
            k = 0             spherical
            k = -1            Paraboloid
            k < -1            Hyperboloid
            -1 < k < 0        Ellipsoid
            k > 0             Oblate eliposid

        References:
            https://www.edmundoptics.com/knowledge-center/application-notes/optics/all-about-aspheric-lenses/

        """
        x0, y0 = r0

        s2 = (self.X - x0)**2 + (self.Y - y0)**2

        t1 = c * s2 / (1 + np.sqrt(1 - (1 + k) * c**2 * s2))

        t2 = 0
        if a is not None:
            for i, ai in enumerate(a):
                t2 = t2 + ai * s2**(2 + i)

        t3 = t1 + t2

        if radius > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, 0)
            t = amplitude.u
        else:
            t = 1

        self.u = t3 * np.exp(1j * 2 * np.pi * (n1 - n0) * t / self.wavelength)

    def lens_cylindrical(self,
                         x0: float,
                         focal: float,
                         refractive_index: float = 1.5,
                         radius: float = 0,
                         angle: float = 0.):
        """Cylindrical lens, without paraxial approximation. The focal distance and the refractive index are used for the definition. When the refractive index decreases, the radius of curvature decrases and less paraxial. When refractive_index is None or 0, then the paraxial approximation is used

        Args:
            x0 (float): center of lens
            focal (float): focal length of lens
            refractive_index (float): refractive index
            radius (float): radius of lens mask

        lens_spherical:
            lens(r0=0, radius= 200 * um, focal= 10 * mm, refractive_index=1.5)
        """

        k = 2 * np.pi / self.wavelength

        r0 = (x0, 0)

        Xrot, Yrot = self.__rotate__(angle, r0)

        if radius > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = 1

        if refractive_index in (None, 0):
            phase = -k * Xrot**2 / (2 * focal)
        else:
            R = (refractive_index - 1) * focal
            h = (np.sqrt(R**2 - Xrot**2) - R)
            h[R**2 - (Xrot**2) < 0] = 0
            phase = k * (refractive_index - 1) * h

        self.u = t * np.exp(1j * phase)

    def fresnel_lens(self,
                     r0: list[floating],
                     focal: floating,
                     levels: list[floating, floating] = (1., 0.),
                     kind: str = 'amplitude',
                     phase: floating = 0.,
                     radius: floating = 0.,
                     angle: floating = 0.):
        """Fresnel lens, amplitude (0,1) or phase (0-phase)

        Args:
            r0 (float, float): (x0,y0) - center of lens
            focal (float, float) or (float): focal length of lens
            radius (float, float) or (float): radius of lens mask
            levels (float, float): levels (1,0) or other of the lens
            kind (str):  'amplitude' or 'phase'
            phase (float): phase shift for phase lens
            angle (float): angle of axis in radians

        Example:
            fresnel_lens( r0=(0 * um, 0 * um), focal=(5 * mm, 10 * mm), radius=200*um, angle=0 * degrees, kind: str = 'amplitude',phase=np.pi)
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)
        if isinstance(focal, (float, int, complex)):
            focal = (focal, focal)

        k = 2 * np.pi / self.wavelength

        f1, f2 = focal

        Xrot, Yrot = self.__rotate__(angle, r0)

        if radius[0] > 0:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t1 = amplitude.u
        else:
            t1 = 1

        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        # t2.u = np.cos(k * ((Xrot**2 / (2 * f1)) + Yrot**2 / (2 * f2)))
        t2.u = np.sin(k * ((Xrot**2 / (2 * f1)) + Yrot**2 / (2 * f2)))

        if kind == 'amplitude':
            t2.u[t2.u > 0] = levels[0]
            t2.u[t2.u <= 0] = levels[1]

        if kind == 'phase':
            t2.u[t2.u > 0] = 1
            t2.u[t2.u <= 0] = 0
            t2.u = np.exp(1j * t2.u * phase)

        self.u = t2.u * t1

    def axicon(self,
               r0: list[floating],
               refractive_index: floating,
               angle: floating,
               radius: floating = 0,
               off_axis_angle: floating = 0 * degrees,
               reflective: bool = False):
        """Axicon,

        Args:
            r0 (float, float): (x0,y0) - center of lens
            refractive_index (float): refractive index
            angle (float): angle of the axicon
            radius (float): radius of lens mask
            off_axis_angle (float) angle when it works off-axis
            reflective (bool): True if the axicon works in reflective mode.

        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0

        # distance de la generatriz al eje del cono
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)

        # Region de transmitancia
        u_mask = np.zeros_like(self.X)
        ipasa = r < radius
        u_mask[ipasa] = 1

        if off_axis_angle == 0 * degrees:
            t_off_axis = 1
        else:
            t_off_axis = np.exp(-1j * k * self.X * np.sin(off_axis_angle))

        if reflective is True:
            self.u = u_mask * np.exp(-2j * k * r * np.tan(angle)) * t_off_axis

        else:
            self.u = u_mask * \
                np.exp(-1j * k * (refractive_index - 1) *
                       r * np.tan(angle)) * t_off_axis

    def axicon_binary(self, r0: list[floating], period: floating, radius: floating = 0.):
        """axicon_binary. Rings with equal period

        Args:
            r0 (float, float): (x0,y0) - center of lens
            radius (float): radius of lens mask
            period (float): distance of rings

        Example:
            axicon_binary(r0=(0 * um, 0 * um),  period=20 * um, radius=200 * um)
        """

        x0, y0 = r0

        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)

        if radius > 0:
            u_mask = np.zeros_like(self.X)
            ipasa = r < radius
            u_mask[ipasa] = 1
        else:
            u_mask = 1

        t = np.cos(2 * np.pi * r / period) * u_mask

        t[t <= 0] = 0
        t[t > 0] = 1

        self.u = t

    def biprism_fresnel(self, r0: list[floating], width: floating, height: floating, n: floating):
        """Fresnel biprism.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            width (float): width
            height (float): height of axicon
            n (float): refractive index

        Example:
            biprism_fresnel(r0=(0 * um, 0 * um), width=100 * \
                            um, height=5 * um, n=1.5)
        """

        k = 2 * np.pi / self.wavelength
        x0, y0 = r0

        xp = self.X > 0
        xn = self.X <= 0

        # Altura desde la base a la surface
        h = np.zeros_like(self.X)
        h[xp] = -2 * height / width * (self.X[xp] - x0) + 2 * height
        h[xn] = 2 * height / width * (self.X[xn] - x0) + 2 * height
        # No existencia de heights negativas
        iremove = h < 0
        h[iremove] = 0

        # Region de transmitancia
        u = np.zeros(np.shape(self.X))
        ipasa = np.abs(self.X - x0) < width
        u[ipasa] = 1

        self.u = u * np.exp(1.j * k * (n - 1) * h)

    def radial_grating(self, r0: list[floating], period: floating, phase: floating, radius: floating,
                       is_binary: bool = True):
        """Radial grating.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled

        Example:
            radial_grating(r0=(0 * um, 0 * um), period=20 * um,
                           phase=0 * um, radius=400 * um, is_binary=True)
        """

        x0, y0 = r0
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        t = 0.5 * (1 + np.sin(2 * np.pi * (r - phase) / period))
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1
        u = np.zeros(np.shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1
        self.u = u * t

    def angular_grating(self, r0: list[floating], num_petals: int, phase: floating, radius: floating,
                        is_binary: bool = True):
        """Angular grating.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled

        Example:
            angular_grating(r0=(0 * um, 0 * um), num_petals =20,
                            phase=0 * um, radius=400 * um, is_binary=True)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos

        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = np.arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = (1 + np.cos((theta - phase) * num_petals)) / 2
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = np.zeros(np.shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def hyperbolic_grating(self,
                           r0: list[floating],
                           period: floating,
                           radius: floating,
                           is_binary: bool,
                           angle: floating = 0.):
        """Hyperbolic grating.

        Args:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled
            angle (float): angle of the grating in radians

        Example:
            hyperbolic_grating(r0=(0 * um, 0 * um), period=20 * \
                               um, radius=400 * um, is_binary=True)
        """

        x0, y0 = r0
        # distance de la generatriz al eje del cono

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        r = np.sqrt((self.X - x0)**2 + (self.Y)**2)
        x_posiciones = np.sqrt(np.abs((Xrot)**2 - (Yrot)**2))

        t = (1 + np.sin(2 * np.pi * x_posiciones / period)) / 2
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = np.zeros(np.shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def hammer(self, r0: list[floating], size: floating, hammer_width: floating, angle: floating = 0.):
        """Square with hammer (like in lithography). Not very useful, an example

        Args:
            r0 (float, float): (x0,y0) - center of square
            size (float, float): size of the square
            hammer_width (float): width of hammer
            angle (float): angle of the grating in radians

        Example:
             hammer(r0=(0 * um, 0 * um), size=(250 * um, 120 * um),
                    hammer_width=5 * um, angle=0 * degrees)
        """
        # si solamente hay 1, las posiciones y radius son los mismos para ambos
        # Origen

        # Definicion del origen y length de la cross

        if len(size) == 1:
            size = (size[0], size[0])

        # Definicion de la cross
        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        th1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        th2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        th3 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        th4 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        # Se define una primera mask cuadrada
        t1.square(r0, size, angle)
        # Una segunda mask cuadrada rotada 90º respecto de la anterior

        # Definicion del square/rectangle
        x0, y0 = r0
        sizex, sizey = size
        xmin = x0 - sizex / 2
        xmax = x0 + sizex / 2
        ymin = y0 - sizey / 2
        ymax = y0 + sizey / 2

        th1.square(r0=(xmin, ymin),
                   size=(hammer_width, hammer_width),
                   angle=angle)
        th2.square(r0=(xmin, ymax),
                   size=(hammer_width, hammer_width),
                   angle=angle)
        th3.square(r0=(xmax, ymin),
                   size=(hammer_width, hammer_width),
                   angle=angle)
        th4.square(r0=(xmax, ymax),
                   size=(hammer_width, hammer_width),
                   angle=angle)
        # La superposicion de ambas da lugar a la cross
        t3 = t1.u + th1.u + th2.u + th3.u + th4.u
        t3[t3 > 0] = 1
        self.u = t3

    def archimedes_spiral(self, r0: list[floating], period: floating, phase: floating, p: int,
                          radius: floating, is_binary: bool):
        """Archimedes spiral

        Args:
            r0 (float, float): (x0,y0) - center of archimedes_spiral
            period (float): period of spiral
            phase (float): initial phase of spiral
            p (int): power of spiral
            radius (float): radius of the mask
            is_binary (bool): if True binary mask

        Example:
            archimedes_spiral(r0=(0 * um, 0 * um), period=20 * degrees,
                              phase=0 * degrees, p=1, radius=200 * um, is_binary=True)
        """

        x0, y0 = r0

        # distance de la generatriz al eje del cono
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = np.arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = 0.5 * (1 + np.sin(2 * np.pi * np.sign(self.X) *
                              ((r / period)**p + (theta - phase) / (2 * np.pi))))
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = np.zeros(np.shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def laguerre_gauss_spiral(self, r0: list[floating], kind: str, n: int, l: int,
                              w0: floating, z: floating):
        """laguerre_gauss spiral

        Args:
            r0 (float, float): (x0,y0) - center of laguerre_gauss_spiral
            kind (str): 'amplitude' or 'phase'
            n (int): of spiral
            l (int): power of spiral
            w0 (float): width of spiral
            z (float): propagation distance

        Example:
            laguerre_gauss_spiral(
                r0=(0 * um, 0 * um), kind='amplitude', l=1, w0=625 * um, z=0.01 * um)
        """

        u_ilum = Scalar_source_XY(x=self.x,
                                  y=self.y,
                                  wavelength=self.wavelength)
        # Haz de Laguerre
        u_ilum.laguerre_beam(A=1, n=n, l=l, r0=r0, w0=w0, z=z, z0=0)

        # Se define el length de la espiral
        length = (self.x.max() - self.x[0]) / 2

        # Se llama a la clase scalar_masks_XY
        t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
        # Hacemos uso de la mask circular
        t1.circle(r0=r0, radius=(length, length), angle=0 * degrees)

        # Se extrae la orientacion de la espiral
        intensity = np.angle(u_ilum.u)
        # Normalizacion
        intensity = intensity / intensity.max()

        # Uso de la mask para obtener la amplitude y la phase
        mask = np.zeros_like(intensity)
        mask[intensity > 0] = 1
        if kind == "phase":
            mask = np.exp(1.j * np.pi * mask)

        self.u = t1.u * mask

    def forked_grating(self, r0: list[floating], period: floating, l: int, alpha: int, kind: str,
                       angle: floating = 0.):
        """Forked grating: np.exp(1.j * alpha * np.cos(l * THETA - 2 * np.pi / period * (Xrot - r0[0])))

        Args:
            r0 (float, float): (x0,y0) - center of forked grating
            period (float): basic period of teh grating
            l (int): *
            alpha (int): *
            kind (str): 'amplitude' or 'phase'
            angle (float): angle of the grating in radians

        Example:
            forked_grating(r0=(0 * um, 0 * um), period=20 * \
                           um, l=2, alpha=1, angle=0 * degrees)
        """
        x0, y0 = r0

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        THETA = np.arctan2(Xrot, Yrot)

        self.u = np.exp(1.j * alpha * np.cos(l * THETA - 2 * np.pi / period * (Xrot)))

        phase = np.angle(self.u)

        phase[phase < 0] = 0
        phase[phase > 0] = 1

        if kind == 'amplitude':
            self.u = phase
        elif kind == 'phase':
            self.u = np.exp(1.j * np.pi * phase)

    def sine_grating(self,
                     x0: float,
                     period: float,
                     amp_min: float = 0,
                     amp_max: float = 1,
                     angle: float = 0.):
        """Sinusoidal grating:  self.u = amp_min + (amp_max - amp_min) * (1 + np.cos(2 * np.pi * (Xrot - phase) / period)) / 2

        Args:
            x0 (float): phase shift
            period (float): period of the grating
            amp_min (float): minimum amplitude
            amp_max (float): maximum amplitud
            angle (float): angle of the grating in radians

        Example:
             sine_grating(period=40 * um, amp_min=0, amp_max=1,
                          x0=0 * um, angle=0 * degrees)
        """
        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        # Definicion de la sinusoidal
        self.u = amp_min + (amp_max -
                            amp_min) * (1 + np.sin(2 * np.pi *
                                                   (Xrot - x0) / period)) / 2

    def sine_edge_grating(self, r0: list[floating], period: floating, lp: floating, ap: floating,
                          phase: floating, radius: floating, is_binary: bool):
        """Generate sin grating

        Args:
            r0 (list[floating]): _description_
            period (floating): _description_
            lp (floating): _description_
            ap (floating): _description_
            phase (floating): _description_
            radius (floating): _description_
            is_binary (bool): _description_
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        # lp longitud del period del edge,
        # ap es la amplitude del period del edge

        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = np.sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        # theta = np.arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        phase_shift = phase + ap * np.sin(2 * np.pi * self.Y / lp)

        t = (1 + np.sin(2 * np.pi * (self.X - phase_shift) / period)) / 2
        if is_binary:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = np.zeros(np.shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def ronchi_grating(self, x0: floating, period: floating, fill_factor: floating = 0.5,
                       angle: floating = 0.):
        """Amplitude binary grating with fill factor: self.u = amp_min + (amp_max - amp_min) * (1 + np.cos(2 * np.pi * (Xrot - phase) / period)) / 2

        Args:
            x0 (float):  phase shift
            period (float): period of the grating
            fill_factor (float): fill_factor
            angle (float): angle of the grating in radians

        Notes:
            Ronchi grating when fill_factor = 0.5.

            It is obtained from a sinusoidal, instead as a sum of slits, for speed.

            The equation to determine the position y0 is: y0=np.cos(np.pi*fill_factor)

        Example:
            ronchi_grating(x0=0 * um, period=40*um, fill_factor=0.5,  angle=0)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        y0 = np.cos(np.pi * fill_factor)

        t.sine_grating(period=period,
                       amp_min=-1,
                       amp_max=1,
                       x0=x0,
                       angle=angle)

        t.u[t.u > y0] = 1
        t.u[t.u <= y0] = 0

        # # Mitad de linea blanca, mitad negra.
        # # Nos quedamos con el valor mayor (e-15) para que en ese tramo valga 1.
        # if ((t.u[0, 0] != t.u[0, -1]) and angle == 90 * degrees):
        #     #print(t.u[0].max())
        #     t.u[0] = t.u[0].max()

        # # Correction 2 (0 degrees)
        # if angle == 0 * degrees:
        #     ind = 0
        #     times = int(2 * t.x.max() / period)
        #     pixel_size = (t.x[1] - t.x[0])
        #     index = np.where(t.u[0, :] == 0)[0]
        #     distancia_minimos = int(period / pixel_size)

        #     for i in range(times - 1):
        #         D_index = index[ind + int(distancia_minimos / 2)] - index[ind]

        #         if D_index != distancia_minimos:
        #             #print('Correcion_Error del periodo')
        #             t.u[:, ind] = 1

        #         ind += int(distancia_minimos / 2)

        self.u = t.u

    def binary_grating(self,
                       x0: float,
                       period: float,
                       fill_factor: float = 0.5,
                       a_min: float = 0,
                       a_max: float = 1,
                       phase: float = 0 * degrees,
                       angle: float = 0.):
        """Binary grating (amplitude and/or phase). The minimum and maximum value of amplitude and phase can be controlled.

         Args:
            x0 (float):  phase shift
            period (float): period of the grating
            fill_factor (float): fill_factor
            a_min (float): minimum amplitude
            a_max (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            angle (float): angle of the grating in radians

        Example:
            binary_grating( x0=0, period=40 * um, fill_factor=0.5,
                           a_min=0, a_max=1, phase=0 * degrees, angle=0 * degrees)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t.ronchi_grating(x0=x0,
                         period=period,
                         fill_factor=fill_factor,
                         angle=angle)
        amplitud = a_min + (a_max - a_min) * t.u
        self.u = amplitud * np.exp(1j * phase * t.u)

    def blazed_grating(self, period: float, phase_max: float, x0: float, angle: float = 0.):
        """Blazed grating.

         Args:
            period (float): period of the grating
            phase_max (float): maximum phase of the blazed grating
            x0 (float): initial displacement of the grating
            angle (float): angle of the grating in radians

        Example:
            blazed_grating(period=40 * um, phase_max=2*np.pi, x0, angle=0 * degrees)
        """
        k = 2 * np.pi / self.wavelength
        # Inclinacion de las franjas
        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        num_periods = (self.x[-1] - self.x[0]) / period

        # Height computation
        phase = (Xrot - x0) * phase_max * num_periods / (self.x[-1] -
                                                         self.x[0])

        # normalization between 0 and 2pi
        phase = np.remainder(phase, phase_max)
        self.u = np.exp(1j * phase)

    def grating_2D(self,
                   r0: list[floating],
                   period: float,
                   fill_factor: float,
                   a_min: float = 0,
                   a_max: float = 1.,
                   phase: float = 0,
                   angle: float = 0.):
        """2D binary grating

         Args:
            r0 (float, r0):  initial position
            period (float, float): period of the grating
            fill_factor (float): fill_factor
            a_min (float): minimum amplitude
            a_max (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            angle (float): angle of the grating in radians

        Example:
            grating_2D(period=40. * um, a_min=0, a_max=1., phase=0. * \
                       np.pi / 2, x0=0, fill_factor=0.75, angle=0.0 * degrees)
        """
        if isinstance(period, (float, int)):
            period = period, period

        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)

        t1.binary_grating(r0[0] + period[0] / 8, period[0], fill_factor, 0, 1,
                          0, angle)
        t2.binary_grating(r0[1] + period[1] / 4, period[1], fill_factor, 0, 1,
                          0, angle + 90. * degrees)

        t2_grating = t1 * t2

        self.u = a_min + (a_max - a_min) * t2_grating.u
        self.u = self.u * np.exp(1j * phase * t2_grating.u)

    def grating_2D_chess(self,
                         r0: list[floating],
                         period: float,
                         fill_factor: float,
                         a_min: float = 0.,
                         a_max: float = 1.,
                         phase: float = 0.,
                         angle: floating = 0.):
        """2D binary grating as chess

         Args:
            r0 (float, r0):  initial position
            period (float): period of the grating
            fill_factor (float): fill_factor
            a_min (float): minimum amplitude
            a_max (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            angle (float): angle of the grating in radians

        Example:
            grating_2D_chess(r0=(0,0), period=40. * um, fill_factor=0.75, a_min=0, a_max=1., phase=0. * \
                             np.pi / 2, angle=0.0 * degrees)
        """

        if isinstance(period, (float, int)):
            period = period, period

        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)

        t1.binary_grating(r0[0], period[0], fill_factor, 0, 1, 0, angle)
        t2.binary_grating(r0[1], period[1], fill_factor, 0, 1, 0,
                          angle + 90. * degrees)

        t2_grating = t1 * t2
        t2_grating.u = np.logical_xor(t1.u, t2.u)

        self.u = a_min + (a_max - a_min) * t2_grating.u
        self.u = self.u * np.exp(1j * phase * t2_grating.u)

    def roughness(self, t: float, s: float, refractive_index: float = -1.):
        """Generation of a rough surface. According to Ogilvy p.224

        Args:
            t (float, float): (tx, ty), correlation length of roughness
            s (float): std of heights
            refractive_index (float): refractive index, if -1 it is reflexion

        Example:
            roughness(t=(50 * um, 25 * um), s=1 * um)
        """

        h_corr = roughness_2D(self.x, self.y, t, s)

        k = 2 * np.pi / self.wavelength
        self.u = np.exp(1.j * k * (refractive_index - 1) * h_corr)
        return h_corr

    def circle_rough(self, r0: list[floating], radius: float, angle: float, sigma: float):
        """Circle with a rough edge.

        Args:
            r0 (float,float): location of center
            radius (float): radius of circle
            angle (float): when radius are not equal, axis of ellipse
            sigma  (float): std of roughness
        """

        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle)

        u = np.zeros(np.shape(self.X))

        random_part = np.random.randn(Yrot.shape[0], Yrot.shape[1])
        ipasa = (Xrot - x0)**2 + (Yrot - y0)**2 - (radius +
                                                   sigma * random_part)**2 < 0
        u[ipasa] = 1
        self.u = u

    def ring_rough(self, r0: list[floating], radius1: float, radius2: float, angle: float, sigma: float):
        """Ring with a rough edge

        Args:
            r0 (float,float): location of center
            radius1 (float): inner radius
            radius2 (float): outer radius
            angle (float): when radius are not equal, axis of ellipse
            sigma  (float): std of roughness
        """

        ring1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring1.circle_rough(r0, radius1, angle, sigma)
        ring2.circle_rough(r0, radius2, angle, sigma)

        # Al restar ring2.u-ring1.u se logra la transmitancia en el interior
        self.u = ring2.u - ring1.u

    def fresnel_lens_rough(self, r0: list[floating], radius: float, focal, angle: float, sigma: float):
        """Ring with a rough edge

        Args:
            r0 (float,float): location of center
            radius (float): maximum radius of mask
            focal (float): outer radius
            angle (float): when radius are not equal, axis of ellipse
            sigma  (float): std of roughness
        """
        lens = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring = Scalar_mask_XY(self.x, self.y, self.wavelength)

        R0 = np.sqrt(self.wavelength * focal)
        num_rings = int(round((radius / R0)**2))

        radius_0 = np.sqrt(self.wavelength * focal * 4) / 2
        ring.circle_rough(r0, radius_0, angle, sigma)
        lens.u = lens.u + ring.u

        for m in range(3, num_rings + 2, 2):
            inner_radius = np.sqrt((m - 1) * self.wavelength * focal)
            outer_radius = np.sqrt(m * self.wavelength * focal)
            ring.ring_rough(r0,
                            inner_radius,
                            outer_radius,
                            angle=angle,
                            sigma=sigma)
            lens.u = lens.u + ring.u
        self.u = lens.u

    def super_ellipse(self, r0: list[floating], radius: float, n: list[int, int] = (2, 2),
                      angle: float = 0.):
        """Super_ellipse. Abs((Xrot - x0) / radiusx)^n1 + Abs()(Yrot - y0) / radiusy)=n2

        Args:
            r0 (float, float): center of super_ellipse
            radius (float, float): radius of the super_ellipse
            n (float, float) =  degrees of freedom of the next equation, n = (n1, n2)
            angle (float): angle of rotation in radians

        Note:
            n1 = n2 = 1: for a square
            n1 = n2 = 2: for a circle
            n1 = n2 = 0.5: for a superellipse

        References:
            https://en.wikipedia.org/wiki/Superellipse


        Example:
            super_ellipse(r0=(0 * um, 0 * um), radius=(250 * \
                          um, 125 * um), angle=0 * degrees)
        """

        if isinstance(r0, (float, int)):
            x0, y0 = (r0, r0)
        else:
            x0, y0 = r0

        if isinstance(n, (int, float)):
            nx, ny = (n, n)
        else:
            nx, ny = n

        assert nx > 0 and ny > 0

        if isinstance(radius, (float, int)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotation of the super-ellipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Definition of transmittance
        u = np.zeros_like(self.X)
        ipasa = np.abs((Xrot) / radiusx)**nx + np.abs((Yrot) / radiusy)**ny < 1
        u[ipasa] = 1
        self.u = u

    def superformula(self, r0: list[floating], radius: float, n: list[int, int, int],
                     m: float, angle: float = 0.):
        """superformula. Abs((Xrot - x0) / radiusx)^n1 + Abs()(Yrot - y0) / radiusy)=n2

        Args:
            r0 (float, float): center of super_ellipse
            radius (float, float): radius of the super_ellipse
            n (float, float, float):  n1, n2, n3 parameters
            m (float): num of petals
            angle (float): angle of rotation in radians


        Note:
            n1 = n2 = 1: for a square
            n1 = n2 = 2: for a circle
            n1 = n2 = 0.5: for a superellipse

        References:
            Gielis, J. "A Generic Geometric Transformation that Unifies a Wide Range of Natural and Abstract Shapes." Amer. J. Botany 90, 333-338, 2003.
            https://mathworld.wolfram.com/Superellipse.html

        Example:
            superformula(r0=(0, 0),  radius=(1.5 * mm, 1.5 * mm), n=(1, 1, 1), m=8, angle=0 * degrees)       
        """

        if isinstance(r0, (float, int)):
            x0, y0 = (r0, r0)
        else:
            x0, y0 = r0

        n1, n2, n3 = n

        if isinstance(radius, (float, int)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotation of the super-ellipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        R = np.sqrt(Xrot**2 + Yrot**2)
        Theta = np.arctan2(Yrot, Xrot)

        # Definition of transmittance
        u = np.zeros_like(self.u)

        factor = max(radiusx, radiusy)

        term1 = np.abs(np.cos(0.25 * m * Theta) / (radiusx / factor))**n2
        term2 = np.abs(np.sin(0.25 * m * Theta) / (radiusy / factor))**n3
        r_theta = (term1 + term2)**(-1 / n1) * factor

        ipasa = R < r_theta

        u[ipasa] = 1
        self.u = u

    def elliptical_phase(self, f1, f2, angle: float):
        """Elliptical phase

        Args:
            f1 (float): focal f1
            f2 (float): focal f2
            angle (float): angle
        """

        k = 2 * np.pi / self.wavelength

        Xrot, Yrot = self.__rotate__(angle)

        phase = k * (Xrot**2 / (2 * f1) + Yrot**2 / (2 * f2))

        self.u = np.exp(1j * phase)

    def sinusoidal_slit(self,
                        size: floating,
                        x0: floating,
                        amplitude: floating,
                        phase: floating,
                        period: floating,
                        angle: floating = 0.):
        """
        This function will create a sinusoidal wave-like slit.

        Args:
            x0 (float): center of slit
            size (float): size of slit
            amplitude (float, float): Phase between the wave-like borders of the slit.
            phase (float): Phase between the wave-like borders of the slit
            period (float): wavelength of the wave-like border of the slit
            angle (float): Angle to be rotated the sinusoidal slit

        Example:
            sinusoidal_slit(y0=(10 * um, -10 * um), amplitude=(10 * um, 20 * um),
                            phase=0 * degrees, angle=0 * degrees, period=(50 * um, 35 * um))
        """

        if isinstance(amplitude, (int, float)):
            amplitude1, amplitude2 = (amplitude, amplitude)
        else:
            amplitude1, amplitude2 = amplitude

        if isinstance(period, (int, float)):
            period1, period2 = (period, period)
        else:
            period1, period2 = period

        assert amplitude1 > 0 and amplitude2 > 0 and period1 > 0 and period2 > 0

        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        u = np.zeros_like(self.X)
        X_sin1 = +size / 2 + amplitude1 * np.sin(2 * np.pi * Yrot / period1)
        X_sin2 = -size / 2 + amplitude2 * np.sin(2 * np.pi * Yrot / period2 +
                                                 phase)
        ipasa_1 = (X_sin1 > Xrot) & (X_sin2 < Xrot)
        u[ipasa_1] = 1
        self.u = u

    def crossed_slits(self, r0: list[floating], slope: floating, angle: floating = 0.):
        """This function will create a crossed slit mask.

        Args:
            r0 (float, float): center of the crossed slit
            slope (float, float): slope of the slit
            angle (float): Angle of rotation of the slit

        Example:
            crossed_slits(r0 = (-10 * um, 20 * um),  slope = 2.5, angle = 30 * degrees)
        """
        if isinstance(slope, (float, int)):
            slope_x, slope_y = (slope, slope)
        else:
            slope_x, slope_y = slope

        if isinstance(r0, (float, int)):
            x0, y0 = (r0, r0)
        else:
            x0, y0 = r0

        # Rotation of the crossed slits
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        u = np.zeros_like(self.X)
        Y1 = slope_x * np.abs(Xrot)  # + y0
        Y2 = slope_y * np.abs(Xrot)  # + y0

        if (slope_x > 0) and (slope_y < 0):
            ipasa = (Yrot > Y1) | (Yrot < Y2)
        elif (slope_x < 0) and (slope_y > 0):
            ipasa = (Yrot < Y1) | (Yrot > Y2)
        elif (slope_x < 0) and (slope_y < 0):
            Y2 = -Y2 + 2 * y0
            ipasa = (Yrot < Y1) | (Yrot > Y2)
        else:
            Y2 = -Y2 + 2 * y0
            ipasa = (Yrot > Y1) | (Yrot < Y2)

        u[ipasa] = 1
        self.u = u

    def hermite_gauss_binary(self, r0: list[floating], w0: list[floating], n: int, m: int):
        """Binary phase mask to generate an Hermite Gauss beam.

        Args:
            r0 (float, float): (x,y) position of source.
            w0 (float, float): width of the beam.
            n (int): order in x.
            m (int): order in y.

        Example:
             hermite_gauss_binary(r0=(0,0), w0=(100*um, 50*um), n=2, m=3)
        """
        # Prepare space
        X = self.X - r0[0]
        Y = self.Y - r0[1]
        r2 = np.sqrt(2)
        wx, wy = w0

        # Calculate amplitude
        E = eval_hermite(n, r2 * X / wx) * eval_hermite(m, r2 * Y / wy)
        phase = np.pi * (E > 0)

        self.u = np.exp(1j * phase)

    def laguerre_gauss_binary(self, r0: list[floating], w0: list[floating], n: floating, l: floating):
        """Binary phase mask to generate an Hermite Gauss beam.

        Args:
            r0 (float, float): (x,y) position of source.
            w0 (float, float): width of the beam.
            n (int): radial order.
            l (int): angular order.

        Example:
             laguerre_gauss_binary(r0=(0,0), w0=1*um, n=0, l=0)
        """
        # Prepare space
        X = self.X - r0[0]
        Y = self.Y - r0[1]
        Ro2 = X**2 + Y**2
        Th = np.arctan2(Y, X)

        # Calculate amplitude
        E = laguerre_polynomial_nk(2 * Ro2 / w0**2, n, l)
        phase = np.pi * (E > 0)

        self.u = np.exp(1j * (phase + l * Th))

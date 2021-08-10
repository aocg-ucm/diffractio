# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_mask_XY class for definingn masks. Its parent is Scalar_field_X.

The main atributes are:
    * self.x - x positions of the field
    * self.z - z positions of the field
    * self.u - field XZ
    * self.n - refraction index XZ
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
    * prism, lens, aspheric, fresnel_lens
    * sine_grating, sine_edge_grating ronchi_grating, binary_grating, blazed_grating, forked_grating, grating2D, grating_2D_chess
    * axicon, biprism_fresnel,
    * radial_grating, angular_grating, hyperbolic_grating, archimedes_spiral, laguerre_gauss_spiral
    * hammer
    * roughness, circle_rough, ring_rough, fresnel_lens_rough,
"""

import matplotlib.figure as mpfig
import matplotlib.image as mpimg
from numpy import (angle, arctan, arctan2, cos, exp, linspace, meshgrid, ones,
                   ones_like, pi, shape, sin, sqrt, zeros, zeros_like)
from PIL import Image
from scipy.signal import fftconvolve
from scipy.special import eval_hermite

from . import degrees, np, plt, sp, um
from .scalar_fields_XY import Scalar_field_XY
from .scalar_sources_XY import Scalar_source_XY
from .utils_math import (fft_convolution2d, laguerre_polynomial_nk, nearest,
                         nearest2)
from .utils_optics import roughness_2D


class Scalar_mask_XY(Scalar_field_XY):
    """Class for working with XY scalar masks.

    Parameters:
        x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n`
        x (numpy.array): linear array with equidistant positions for y values
        wavelength (float): wavelength of the incident field
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions. The number of data is preferibly :math:`2^n` .
        self.y (numpy.array): linear array wit equidistant positions for y values
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): (x,z) complex field
        self.info (str): String with info about the simulation
    """

    def __init__(self, x=None, y=None, wavelength=None, info=""):
        # print("init de Scalar_mask_XY")
        super(self.__class__, self).__init__(x, y, wavelength, info)
        self.type = 'Scalar_mask_XY'

    def set_amplitude(self, q=1, positive=0, amp_min=0, amp_max=1):
        """makes that the mask has only amplitude.

        Parameters:
            q (int): 0 - amplitude as it is and phase is removed. 1 - take phase and convert to amplitude

            positive (int): 0 - value may be positive or negative. 1 - value is only positive
        """

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        if q == 0:
            if positive == 0:
                self.u = amp_min + (amp_max -
                                    amp_min) * amplitude * np.sign(phase)
            if positive == 1:
                self.u = amp_min + (amp_max - amp_min) * amplitude
        else:
            if positive == 0:
                self.u = amp_min + (amp_max - amp_min) * phase
            if positive == 1:
                self.u = amp_min + (amp_max - amp_min) * np.abs(phase)

        # hay que terminar

    def set_phase(self, q=1, phase_min=0, phase_max=pi):
        """Obliga a la mask a ser de phase,
            q=0: toma la phase que hay y hace la amplitude=1
            q=1: la amplitude la pasa a phase"""

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        if q == 0:
            self.u = exp(1.j * phase)
        if q == 1:
            self.u = exp(1.j * (phase_min +
                                (phase_max - phase_min) * amplitude))

    def area(self, percentaje):
        """Computes area where mask is not 0

        Parameters:
            percentaje_maximum (float): percentaje from maximum intensity to compute

        Returns:
            float: area (in um**2)

        Example:
            area(percentaje=0.001)
        """

        intensity = np.abs(self.u)**2
        max_intensity = intensity.max()
        num_pixels_1 = sum(sum(intensity > max_intensity * percentaje))
        num_pixels = len(self.x) * len(self.y)
        delta_x = self.x[1] - self.x[0]
        delta_y = self.y[1] - self.y[0]

        return (num_pixels_1 / num_pixels) * (delta_x * delta_y)

    def inverse_amplitude(self):
        """Inverts the amplitude of the mask, phase is equal as initial"""
        amplitude = np.abs(self.u)
        phase = angle(self.u)

        self.u = (1 - amplitude) * exp(1.j * phase)

    def inverse_phase(self):
        """Inverts the phase of the mask, amplitude is equal as initial"""
        amplitude = np.abs(self.u)
        phase = angle(self.u)

        self.u = amplitude * exp(-1.j * phase)

    def filter(self, mask, new_field=True, binarize=False, normalize=False):
        """Widens a field using a mask

        Parameters:
            mask (diffractio.Scalar_mask_X): filter
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

    def widen(self, radius, new_field=True, binarize=True):
        """Widens a mask using a convolution of a certain radius

        Parameters:
            radius (float): radius of convolution
            new_field (bool): returns a new XY field
            binarize (bool): binarizes result.
        """

        filter = Scalar_mask_XY(self.x, self.y, self.wavelength)
        filter.circle(r0=(0 * um, 0 * um),
                      radius=(radius, radius),
                      angle=0 * degrees)

        image = np.abs(self.u)
        filtrado = np.abs(filter.u) / np.abs(filter.u.sum())

        covolved_image = fft_convolution2d(image, filtrado)
        average = (covolved_image.max()) / 2
        if binarize is True:
            covolved_image[covolved_image > average] = 1
            covolved_image[covolved_image <= average] = 0

        if new_field is True:
            filter.u = covolved_image
            return filter
        else:
            self.u = covolved_image

    # __MASCARAS PROPIAMENTE DICHAS____________________________________________

    def extrude_mask_x(self,
                       mask_X,
                       y0=None,
                       y1=None,
                       kind='unique',
                       normalize=None):
        """
        Converts a Scalar_mask_X in volumetric between z0 and z1 by growing between these two planes
        Parameters:
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

        iy0, value, distance = nearest(vector=self.y, number=y0)
        iy1, value, distance = nearest(vector=self.y, number=y1)

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

    def mask_from_function(self,
                           r0,
                           index,
                           f1,
                           f2,
                           radius,
                           v_globals={},
                           mask=True):
        """ phase mask defined between 2 surfaces $f_1$ and $f_2$:  $h(x,y)=f_2(x,y)-f_1(x,y)$

        Parameters:
            r0 (float, float): center of cross
            index (float): refraction index
            f1 (str): function for first surface
            f2 (str): function for second surface
            radius (float, float) or (float): size of mask
            v_globals (dict): dictionary with globals
            mask (bool): If True applies mask
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        k = 2 * pi / self.wavelength

        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, 0 * degrees)
            t = amplitude.u
        else:
            t = ones_like(self.X)

        v_locals = {'self': self, 'sp': sp, 'degrees': degrees}

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)
        self.u = t * exp(1.j * k * (index - 1) * (F2 - F1))
        self.u[t == 0] = 0

    def image(self,
              filename='',
              canal=0,
              normalize=True,
              lengthImage=False,
              invert=False,
              angle=0):
        """Converts an image file XY mask. If the image is color, we get the first Red frame

        Parameters:
            filename (str): filename of the image
            canal (int): number of channel RGB to get the image
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

        # Seleccionamos un canal de color
        image = colores[canal]

        # data = image.getdata()

        # Reajuste del length manteniendo la relacion de aspecto
        if lengthImage is False:
            length = self.u.shape
            image = image.resize(length)

        if lengthImage is True:
            length = im.size
            self.x = linspace(self.x[0], self.x[-1], length[0])
            self.y = linspace(self.y[0], self.y[-1], length[1])
            self.X, self.Y = meshgrid(self.x, self.y)

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

    def image2(self, filename, negativo=True):
        imagen1 = mpimg.imread(filename)
        imgshow = plt.imshow(
            imagen1,
            vmin=0,
            vmax=1,
            aspect='auto',
            extent=[self.x.min(),
                    self.x.max(),
                    self.y.min(),
                    self.y.max()])
        imagen1 = mpfig.Figure()
        imgshow.set_cmap('hot')

        T = Scalar_mask_XY(self.x, self.y, self.wavelength)
        T.u = imagen1
        u = np.zeros_like(self.X)
        u = u + imagen1

        self.u = 1 - u

    def triangle(self, r0=None, slope=2.0, height=50 * um, angle=0 * degrees):
        """Create a triangle mask. It uses the equation of a straight line: y = -slope * (x - x0) + y0

        Parameters:
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

    def photon_sieve(self, t1, r0):
        """Generates a matrix of shapes given in t1.

        Parameters:
            t1 (Scalar_mask_XY): Mask of the desired figure to be drawn
            r0 (float, float) or (np.array, np.array): (x,y) point or points where mask is 1


        Returns:
            (int): number of points in the mask

        """

        x0, y0 = r0
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
        u[u > 1] = 1
        self.u = u
        return num_points

    def insert_array_masks(self, t1, space, margin=0, angle=0 * degrees):
        """Generates a matrix of shapes given in t1.

        Parameters:
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

        Xrot, Yrot = self.__rotate__(angle)

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

    def dots(self, r0):
        """Generates 1 or several point masks at positions r0

        Parameters:
            r0 (float, float) or (np.array, np.array): (x,y) point or points where mask is 1


        """
        x0, y0 = r0
        u = np.zeros_like(self.X)

        if type(r0[0]) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            i_y0, _, _ = nearest(self.y, y0)
            u[i_x0, i_y0] = 1
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            i_y0s, _, _ = nearest2(self.y, y0)
            for (i_x0, i_y0) in zip(i_x0s, i_y0s):
                u[i_x0, i_y0] = 1

        self.u = u
        return self

    def dots_regular(self, xlim, ylim, num_data, verbose=False):
        """Generates n x m or several point masks.

        Parameters:
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
        return self

    def one_level(self, level=0):
        """Sets one level for all the image.

        Parameters:
            level (float): value
        """
        self.u = level * ones(self.X.shape)

    def two_levels(self, level1=0, level2=1, x_edge=0, angle=0):
        """Divides the field in two levels

        Parameters:
            level1 (float): value of first level
            level2 (float): value of second level
            x_edge (float): position of division
            angle (float): angle of rotation in radians
        """
        Xrot, Yrot = self.__rotate__(angle, (x_edge, 0))
        self.u = level1 * ones(self.X.shape)
        self.u[Xrot > 0] = level2

    def edge_series(self,
                    r0,
                    period,
                    a_coef,
                    b_coef=None,
                    angle=0 * degrees,
                    invert=True):
        """Creates a linear aperture using the Fourier coefficients.

            Parameters:
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

    def slit(self, x0, size, angle=0 * degrees):
        """Slit: 1 inside, 0 outside

        Parameters:
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
        u = zeros(shape(self.X))
        ix = (Xrot < xmax) & (Xrot > xmin)
        u[ix] = 1
        self.u = u

    def slit_series(self,
                    x0,
                    width,
                    period1,
                    period2,
                    Dy,
                    a_coef1,
                    a_coef2,
                    b_coef1=None,
                    b_coef2=None,
                    angle=None,
                    simmetrycal=False):
        """Creates a lineal function using the Fourier coefficients.

            Parameters:
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
                simmetrical (bool): TODO - take edge 1 and repeat simmetrical y 2

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

    def double_slit(self, x0, size, separation, angle=0 * degrees):
        """double slit: 1 inside, 0 outside

        Parameters:
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

    def square(self, r0, size, angle):
        """Square: 1 inside, 0 outside

        Parameters:
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
        u = zeros(shape(self.X))
        ipasa = (Xrot < xmax) & (Xrot > xmin) & (Yrot < ymax) & (Yrot > ymin)
        u[ipasa] = 1
        self.u = u

    def gray_scale(self, num_levels=4, levelMin=0, levelMax=1):
        """Generates a number of strips with different amplitude

        Parameters:
            num_levels (int): number of levels
            levelMin (float): value of minimum level
            levelMax (float): value of maximum level
        """
        t = zeros(self.X.shape, dtype=float)

        xpos = linspace(self.x[0], self.x[-1], num_levels + 1)
        height_levels = linspace(levelMin, levelMax, num_levels)
        ipos, _, _ = nearest2(self.x, xpos)
        ipos[-1] = len(self.x)
        # print(ipos)

        for i in range(num_levels):
            # print(ipos[i + 1], ipos[i])
            t[:, ipos[i]:ipos[i + 1]] = height_levels[i]

        self.u = t

    def circle(self, r0, radius, angle=0 * degrees):
        """Creates a circle or an ellipse.

        Parameters:
            r0 (float, float): center of circle/ellipse
            radius (float, float) or (float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            circle(r0=(0 * um, 0 * um), radius=(250 * \
                   um, 125 * um), angle=0 * degrees)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Definicion de la transmitancia
        u = zeros(shape(self.X))
        ipasa = (Xrot)**2 / (radiusx + 1e-15)**2 + \
            (Yrot)**2 / (radiusy**2 + 1e-15) < 1
        u[ipasa] = 1
        self.u = u

    def super_gauss(self, r0, radius, potencia=2, angle=0 * degrees):
        """Supergauss mask.

        Parameters:
            r0 (float, float): center of circle
            radius (float, float) or (float): radius of circle
            potencia (float): value of exponential
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
        R = sqrt(self.X**2 + self.Y**2)
        self.u = exp(-R**potencia / (2 * radiusx**potencia))

    def square_circle(self, r0, R1, R2, s, angle=0 * degrees):
        """ Between circle and square, depending on fill factor s

        s=0 circle, s=1 square

        Parameters:
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
        F = sqrt(Xrot**2 / R1**2 + Yrot**2 / R2**2 - s**2 * Xrot**2 * Yrot**2 /
                 (R1**2 * R2**2))

        Z1 = F < 1
        Z = Z1 * t1.u

        self.u = Z

    def angular_aperture(self, a_coef, b_coef=None, angle=0 * degrees):
        """Creates a radial function using the Fourier coefficients.

            Parameters:

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

    def ring(self, r0, radius1, radius2, angle=0 * degrees):
        """ Ring.

        Parameters:
            r0 (float, float): center of ring
            radius1 (float, float) or (float): inner radius
            radius2 (float, float) or (float): outer radius
            angle (float): angle of rotation in radians
        """

        ring1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring1.circle(r0, radius1, angle)
        ring2.circle(r0, radius2, angle)

        self.u = ring2.u - ring1.u

    def rings(self, r0, inner_radius, outer_radius, mask=True):
        """Structure based on several rings, with radius given by inner_radius and outer_radius.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            inner_radius (np.array): inner radius
            outer_radius (np.array): inner radius
            mask (bool): if True, mask with size radius of maximum outer radius
        """

        x0, y0 = r0
        angle = 0

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        radius = outer_radius.max()

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = ones_like(self.X)

        u = np.zeros_like(self.X)
        ring = Scalar_mask_XY(self.x, self.y, self.wavelength)

        num_rings = len(inner_radius)

        for i in range(num_rings):
            ring.ring(r0, inner_radius[i], outer_radius[i], angle)
            u = u + ring.u

        self.u = u

        self.u[t == 0] = 0
        return self

    def cross(self, r0, size, angle=0 * degrees):
        """ Cross

        Parameters:
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

    def prism(self, r0, angle_wedge, angle=0 * degrees):
        """prism which produces a certain angle

        Parameters:
            r0 (float, float): center wedge
            angle_wedge (float): angle of wedge in x direction
            angle (float): angle of rotation in radians

        """
        # Vector de onda
        k = 2 * pi / self.wavelength
        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        self.u = exp(1j * k * (Xrot) * np.sin(angle_wedge))

    def lens(self, r0, radius, focal, angle=0 * degrees, mask=True):
        """Transparent lens

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float) or (float): radius of lens mask
            focal (float, float) or (float): focal length of lens
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius

        Example:
            lens(r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True)
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)
        if isinstance(focal, (float, int, complex)):
            focal = (focal, focal)

        # Vector de onda
        k = 2 * pi / self.wavelength

        x0, y0 = r0
        f1, f2 = focal

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t = amplitude.u
        else:
            t = ones_like(self.X)

        self.u = t * exp(-1.j * k * ((Xrot**2 / (2 * f1)) + Yrot**2 /
                                     (2 * f2)))
        self.u[t == 0] = 0

    def aspheric(self, r0, c, k, a, n0, n1, radius, mask=True):
        """asferic surface.

        $z = \frac{c r^2}{1+\sqrt{1-(1+k) c^2 r^2 }}+\sum{a_i r^{2i}}$

        Parameters:
            x0 (float): position of center
            c (float): base curvature at vertex, inverse of radius
            k (float): conic constant
            a (list): order aspheric coefficients: A4, A6, A8, ...
            n0 (float): refraction index of first medium
            n1 (float): refraction index of second medium
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
                t2 = t2 + ai * s2 ** (2 + i)

        t = t1 + t2

        if mask is True:
            m1 = np.zeros_like(self.x, dtype=int)
            ix = (self.x < x0 + radius) & (self.x > x0 - radius)
            m1[ix] = 1
        else:
            m1 = np.ones_like(self.x, dtype=int)

        self.u = m1 * np.exp(1j * 2 * np.pi * (n1 - n0) * t / self.wavelength)
        self.u[m1 == 0] = 0
        return t

    def fresnel_lens(self,
                     r0,
                     radius,
                     focal,
                     levels=(1, 0),
                     kind='amplitude',
                     phase=pi,
                     angle=0 * degrees,
                     mask=True):
        """Fresnel lens, amplitude (0,1) or phase (0-phase)

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float) or (float): radius of lens mask
            focal (float, float) or (float): focal length of lens
            levels (float, float): levels (1,0) or other of the lens
            kind (str):  'amplitude' or 'phase'
            phase (float): phase shift for phase lens
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius

        Example:
            fresnel_lens( r0=(0 * um, 0 * um), radius=(100 * um, 200 * um), focal=(5 * mm, 10 * mm), angle=0 * degrees, mask=True, kind='amplitude',phase=pi)
        """

        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)
        if isinstance(focal, (float, int, complex)):
            focal = (focal, focal)

        # Vector de onda
        k = 2 * pi / self.wavelength

        x0, y0 = r0
        f1, f2 = focal

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle, r0)

        # Definicion de la amplitude y la phase
        if mask is True:
            amplitude = Scalar_mask_XY(self.x, self.y, self.wavelength)
            amplitude.circle(r0, radius, angle)
            t1 = amplitude.u
        else:
            t1 = ones_like(self.X)

        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2.u = cos(k * ((Xrot**2 / (2 * f1)) + Yrot**2 / (2 * f2)))
        t2.u[t2.u > 0] = levels[0]
        t2.u[t2.u <= 0] = levels[1]

        if kind == 'phase':
            t2.u = exp(1j * t2.u * phase)

        self.u = t2.u * t1

    def axicon(self,
               r0,
               radius,
               angle,
               refraction_index,
               off_axis_angle=0 * degrees,
               reflective=False):
        """Axicon,

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float): radius of lens mask
            height (float): height of axicon
            n (float): refraction index

        Example:
            axicon(r0=(0 * um, 0 * um), radius=200 * um, height=5 * um,  n=1.5)
        """
        # Vector de onda
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
                np.exp(-1j * k * (refraction_index - 1)
                       * r * np.tan(angle)) * t_off_axis

    def biprism_fresnel(self, r0, width, height, n):
        """Fresnel biprism.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            width (float): width
            height (float): height of axicon
            n (float): refraction index

        Example:
            biprism_fresnel(r0=(0 * um, 0 * um), width=100 * um, height=5 * um, n=1.5)
        """

        # Vector de onda
        k = 2 * pi / self.wavelength
        x0, y0 = r0

        xp = self.X > 0
        xn = self.X <= 0

        # Altura desde la base a la surface
        h = zeros_like(self.X)
        h[xp] = -2 * height / width * (self.X[xp] - x0) + 2 * height
        h[xn] = 2 * height / width * (self.X[xn] - x0) + 2 * height
        # No existencia de heights negativas
        iremove = h < 0
        h[iremove] = 0

        # Region de transmitancia
        u = zeros(shape(self.X))
        ipasa = np.abs(self.X - x0) < width
        u[ipasa] = 1

        self.u = u * exp(1.j * k * (n - 1) * h)

    def radial_grating(self, r0, period, phase, radius, is_binary=True):
        """Radial grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled

        Example:
            radial_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, radius=400 * um, is_binary=True)
        """

        x0, y0 = r0
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        t = 0.5 * (1 + sin(2 * pi * (r - phase) / period))
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1
        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1
        self.u = u * t

    def angular_grating(self, r0, period, phase, radius, is_binary=True):
        """Angular grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled

        Example:
            angular_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, radius=400 * um, is_binary=True)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos

        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = (1 + sin(2 * pi * (theta - phase) / period)) / 2
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def hyperbolic_grating(self,
                           r0,
                           period,
                           phase,
                           radius,
                           is_binary,
                           angle=0 * degrees):
        """Hyperbolic grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            is_binary (bool): if True binary else, scaled
            angle (float): angle of the grating in radians

        Example:
            hyperbolic_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, sfradius=400 * um, is_binary=True)
        """

        x0, y0 = r0
        # distance de la generatriz al eje del cono

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        r = sqrt((self.X - x0)**2 + (self.Y)**2)
        x_posiciones = sqrt(np.abs((Xrot)**2 - (Yrot)**2))
        # Region de transmitancia
        t = (1 + sin(2 * pi * x_posiciones / period)) / 2
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def hammer(self, r0, size, hammer_width, angle=0 * degrees):
        """Square with hammer (like in lithography). Not very useful, an example

        Parameters:
            r0 (float, float): (x0,y0) - center of square
            size (float, float): size of the square
            hammer_width (float): width of hammer
            angle (float): angle of the grating in radians

        Example:
             hammer(r0=(0 * um, 0 * um), size=(250 * um, 120 * um), hammer_width=5 * um, angle=0 * degrees)
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

    def archimedes_spiral(self, r0, period, phase, p, radius, is_binary):
        """Archimedes spiral

        Parameters:
            r0 (float, float): (x0,y0) - center of archimedes_spiral
            period (float): period of spiral
            phase (float): initial phase of spiral
            p (int): power of spiral
            radius (float): radius of the mask
            is_binary (bool): if True binary mask

        Example:
            archimedes_spiral(r0=(0 * um, 0 * um), period=20 * degrees, phase=0 * degrees, p=1, radius=200 * um, is_binary=True)
        """

        x0, y0 = r0

        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = 0.5 * (1 + sin(2 * pi * np.sign(self.X) *
                           ((r / period)**p + (theta - phase) / (2 * pi))))
        if is_binary is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def laguerre_gauss_spiral(self, r0, kind, n, l, w0, z):
        """laguerre_gauss spiral

        Parameters:
            r0 (float, float): (x0,y0) - center of laguerre_gauss_spiral
            kind (str): 'amplitude' or 'phase'
            n (int): of spiral
            l (int): power of spiral
            w0 (float): width of spiral
            z (float): propagation distance

        Example:
            laguerre_gauss_spiral(r0=(0 * um, 0 * um), kind='amplitude', l=1, w0=625 * um, z=0.01 * um)
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
        intensity = angle(u_ilum.u)
        # Normalizacion
        intensity = intensity / intensity.max()

        # Uso de la mask para obtener la amplitude y la phase
        mask = zeros_like(intensity)
        mask[intensity > 0] = 1
        if kind == "phase":
            mask = exp(1.j * pi * mask)

        self.u = t1.u * mask

    def forked_grating(self, r0, period, l, alpha, kind, angle=0 * degrees):
        """Forked grating: exp(1.j * alpha * cos(l * THETA - 2 * pi / period * (Xrot - r0[0])))

        Parameters:
            r0 (float, float): (x0,y0) - center of forked grating
            period (float): basic period of teh grating
            l (int): *
            alpha (int): *
            kind (str): 'amplitude' or 'phase'
            angle (float): angle of the grating in radians

        Example:
            forked_grating(r0=(0 * um, 0 * um), period=20 * um, l=2, alpha=1, angle=0 * degrees)
        """
        x0, y0 = r0

        Xrot, Yrot = self.__rotate__(angle, (x0, y0))

        THETA = arctan2(Xrot, Yrot)

        self.u = exp(1.j * alpha * cos(l * THETA - 2 * pi / period * (Xrot)))

        phase = np.angle(self.u)

        phase[phase < 0] = 0
        phase[phase > 0] = 1

        if kind == 'amplitude':
            self.u = phase
        elif kind == 'phase':
            self.u = exp(1.j * pi * phase)

    def sine_grating(self,
                     x0,
                     period,
                     amp_min=0,
                     amp_max=1,
                     angle=0 * degrees):
        """Sinusoidal grating:  self.u = amp_min + (amp_max - amp_min) * (1 + cos(2 * pi * (Xrot - phase) / period)) / 2

        Parameters:
            x0 (float): phase shift
            period (float): period of the grating
            amp_min (float): minimum amplitude
            amp_max (float): maximum amplitud
            angle (float): angle of the grating in radians

        Example:
             sine_grating(period=40 * um, amp_min=0, amp_max=1, x0=0 * um, angle=0 * degrees)
        """
        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        # Definicion de la sinusoidal
        self.u = amp_min + (amp_max -
                            amp_min) * (1 + sin(2 * pi *
                                                (Xrot - x0) / period)) / 2

    def sine_edge_grating(self, r0, period, lp, ap, phase, radius, is_binary):
        """
        TODO: function info
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        # lp longitud del period del edge,
        # ap es la amplitude del period del edge

        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        # theta = arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        Desphase = phase + ap * sin(2 * pi * self.Y / lp)

        t = (1 + sin(2 * pi * (self.X - Desphase) / period)) / 2
        if is_binary:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def ronchi_grating(self, x0, period, fill_factor=0.5, angle=0):
        """Amplitude binary grating with fill factor: self.u = amp_min + (amp_max - amp_min) * (1 + cos(2 * pi * (Xrot - phase) / period)) / 2

        Parameters:
            x0 (float):  phase shift
            period (float): period of the grating
            fill_factor (float): fill_factor
            angle (float): angle of the grating in radians

        Notes:
            Ronchi grating when fill_factor = 0.5.

            It is obtained from a sinusoidal, instead as a sum of slits, for speed.

            The equation to determine the position y0 is: y0=cos(pi*fill_factor)

        Example:
            ronchi_grating(x0=0 * um, period=40*um, fill_factor=0.5,  angle=0)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        y0 = cos(pi * fill_factor)

        t.sine_grating(period=period,
                       amp_min=-1,
                       amp_max=1,
                       x0=x0,
                       angle=angle)
        t.u[t.u > y0] = 1
        t.u[t.u <= y0] = 0
        self.u = t.u

    def binary_grating(self,
                       x0,
                       period,
                       fill_factor=0.5,
                       amin=0,
                       amax=1,
                       phase=0 * degrees,
                       angle=0 * degrees):
        """Binary grating (amplitude and/or phase). The minimum and maximum value of amplitude and phase can be controlled.

         Parameters:
            period (float): period of the grating
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            x0 (float):  phase shift
            fill_factor (float): fill_factor
            angle (float): angle of the grating in radians

        Example:
            binary_grating( x0=0, period=40 * um, fill_factor=0.5, amin=0, amax=1, phase=0 * degrees, angle=0 * degrees)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t.ronchi_grating(period=period,
                         fill_factor=fill_factor,
                         x0=x0,
                         angle=angle)
        amplitud = amin + (amax - amin) * t.u
        self.u = amplitud * np.exp(1j * phase * t.u)

    def blazed_grating(self, period, height, index, x0, angle=0 * degrees):
        """Binary grating (amplitude and/or phase). The minimum and maximum value of amplitude and phase can be controlled.

         Parameters:
            period (float): period of the grating
            height (float): height of the blazed grating
            index (float): refraction index
            x0 (float): initial displacement of the grating
            angle (float): angle of the grating in radians

        Example:
            blazed_grating(period=40 * um, height=2 * um, index=1.5, x0, angle=0 * degrees)
        """
        k = 2 * pi / self.wavelength
        # Inclinacion de las franjas
        Xrot, Yrot = self.__rotate__(angle, (x0, 0))

        # Calculo de la pendiente
        pendiente = height / period
        # Calculo de la height
        h = (Xrot) * pendiente

        # Calculo del a phase
        phase = k * (index - 1) * h
        # Definicion del origen
        phase = phase - phase.min()
        # Normalizacion entre 0 y 2pi
        phase = np.remainder(phase, 2 * pi)
        self.u = exp(1j * phase)

    def grating_2D(self,
                   r0,
                   period,
                   fill_factor,
                   amin=0,
                   amax=1.,
                   phase=0,
                   angle=0 * degrees):
        """2D binary grating

         Parameters:
            r0 (float, r0):  initial position
            period (float, float): period of the grating
            fill_factor (float): fill_factor
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            angle (float): angle of the grating in radians

        Example:
            grating_2D(period=40. * um, amin=0, amax=1., phase=0. * pi / 2, x0=0, fill_factor=0.75, angle=0.0 * degrees)
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

        self.u = amin + (amax - amin) * t2_grating.u
        self.u = self.u * np.exp(1j * phase * t2_grating.u)

    def grating_2D_chess(self,
                         r0,
                         period,
                         fill_factor,
                         amin=0,
                         amax=1,
                         phase=0 * pi / 2,
                         angle=0 * degrees):
        """2D binary grating as chess

         Parameters:
            r0 (float, r0):  initial position
            period (float): period of the grating
            fill_factor (float): fill_factor
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            angle (float): angle of the grating in radians

        Example:
            grating_2D_chess(period=40. * um, amin=0, amax=1., phase=0. * pi / 2, x0=0, fill_factor=0.75, angle=0.0 * degrees)
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
        t2_grating.u = np.logical_xor(t1.u, t2.u)

        self.u = amin + (amax - amin) * t2_grating.u
        self.u = self.u * np.exp(1j * phase * t2_grating.u)

    def roughness(self, t, s):
        """Generation of a rough surface. According to Ogilvy p.224

        Parameters:
            t (float, float): (tx, ty), correlation length of roughness
            s (float): std of heights

        Example:
            roughness(t=(50 * um, 25 * um), s=1 * um)
        """

        h_corr = roughness_2D(self.x, self.y, t, s)

        k = 2 * pi / self.wavelength
        self.u = exp(-1.j * k * 2 * h_corr)
        return h_corr

    def circle_rough(self, r0, radius, angle, sigma, correlation_length):
        """Circle with a rough edge.

        Parameters:
            r0 (float,float): location of center
            radius (float): radius of circle
            angle (float): when radius are not equal, axis of ellipse
            sigma  (float): std of roughness
            correlation_length (float): correlation length of roughness
        """

        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle)

        u = zeros(shape(self.X))

        random_part = np.random.randn(Yrot.shape[0], Yrot.shape[1])
        ipasa = (Xrot - x0)**2 + (Yrot - y0)**2 - (radius +
                                                   sigma * random_part)**2 < 0
        u[ipasa] = 1
        self.u = u

    def ring_rough(self, r0, radius1, radius2, angle, sigma,
                   correlation_length):
        """Ring with a rough edge

        Parameters:
            r0 (float,float): location of center
            radius1 (float): inner radius
            radius2 (float): outer radius
            angle (float): when radius are not equal, axis of ellipse
            sigma  (float): std of roughness
            correlation_length (float): correlation length of roughness
        """

        ring1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring1.circle_rough(r0, radius1, angle, sigma, correlation_length)
        ring2.circle_rough(r0, radius2, angle, sigma, correlation_length)

        # Al restar ring2.u-ring1.u se logra la transmitancia en el interior
        self.u = ring2.u - ring1.u

    def fresnel_lens_rough(self, r0, radius, focal, angle, mask, kind, phase,
                           sigma, correlation_length):
        """Ring with a rough edge

        Parameters:
            r0 (float,float): location of center
            radius (float): maximum radius of mask
            focal (float): outer radius
            angle (float): when radius are not equal, axis of ellipse
            mask (bool):
            kind (str): 'amplitude' o 'phase'
            phase (float): maximum phase shift, only if kind='phase'
            sigma  (float): std of roughness
            correlation_length (float): correlation length of roughness
        """
        lens = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring = Scalar_mask_XY(self.x, self.y, self.wavelength)

        R0 = sqrt(self.wavelength * focal)
        num_rings = int(round((radius / R0)**2))

        radius_0 = sqrt(self.wavelength * focal * 4) / 2
        ring.circle_rough(r0, radius_0, angle, sigma, correlation_length)
        lens.u = lens.u + ring.u

        for m in range(3, num_rings + 2, 2):
            inner_radius = sqrt((m - 1) * self.wavelength * focal)
            outer_radius = sqrt(m * self.wavelength * focal)
            ring.ring_rough(r0,
                            inner_radius,
                            outer_radius,
                            angle=angle,
                            sigma=sigma,
                            correlation_length=correlation_length)
            lens.u = lens.u + ring.u
        self.u = lens.u

    def super_ellipse(self, r0, radius, n=(2, 2), angle=0 * degrees):
        """Super_ellipse. Abs((Xrot - x0) / radiusx)^n1 + Abs()(Yrot - y0) / radiusy)=n2

        Parameters:
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
            super_ellipse(r0=(0 * um, 0 * um), radius=(250 * um, 125 * um), angle=0 * degrees)
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
        ipasa = np.abs((Xrot) / radiusx)**nx + \
            np.abs((Yrot) / radiusy)**ny < 1
        u[ipasa] = 1
        self.u = u

    def elliptical_phase(self, f1, f2, angle):
        """Elliptical phase

        Parameters:
            f1 (float): focal f1
            f2 (float): focal f2
            angle (float): angle
        """

        # Vector de onda
        k = 2 * pi / self.wavelength

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        phase = k * (Xrot**2 / (2 * f1) + Yrot**2 / (2 * f2))

        self.u = np.exp(1j * phase)

    def sinusoidal_slit(self,
                        size,
                        x0,
                        amplitude,
                        phase,
                        period,
                        angle=0 * degrees):
        """
        This function will create a sinusoidal wave-like slit.

        Parameters:
            x0 (float): center of slit
            size (float): size of slit
            amplitude (float, float): Phase between the wave-like borders of the slit.
            phase (float): Phase between the wave-like borders of the slit
            period (float): wavelength of the wave-like border of the slit
            angle (float): Angle to be rotated the sinusoidal slit

        Example:
            sinusoidal_slit(y0=(10 * um, -10 * um), amplitude=(10 * um, 20 * um), phase=0 * degrees, angle=0 * degrees, period=(50 * um, 35 * um))
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

    def crossed_slits(self, r0, slope, angle=0 * degrees):
        """This function will create a crossed slit mask.

        Parameters:
            r0 (float, float): center of the crossed slit
            slope (float, float): slope of the slit
            angle (float): Angle of rotation of the slit

        Example:
            crossed_slits(r0 = (-10 * um, 20 * um), slope = 2.5, angle = 30 * degrees)
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

    def hermite_gauss_binary(self, r0, w0, n, m):
        """Binary phase mask to generate an Hermite Gauss beam.

        Parameters:
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
        r2 = sqrt(2)
        wx, wy = w0

        # Calculate amplitude
        E = eval_hermite(n, r2 * X / wx) * eval_hermite(m, r2 * Y / wy)
        phase = pi * (E > 0)

        self.u = exp(1j * phase)

    def laguerre_gauss_binary(self, r0, w0, n, l):
        """Binary phase mask to generate an Hermite Gauss beam.

        Parameters:
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
        phase = pi * (E > 0)

        self.u = exp(1j * (phase + l * Th))

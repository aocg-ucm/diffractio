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
    * slit, double_slit, square, circle, super_gauss, square_circle, ring, cross
    * mask_from_function
    * prism, lens, fresnel_lens, lens_billet,
    * sine_grating, sine_edge_grating ronchi_grating, binary_grating, blazed_grating, forked_grating, grating2D, grating_2D_chess
    * axicon, biprism_fresnel,
    * radial_grating, angular_grating, hyperbolic_grating, archimedes_spiral, laguerre_gauss_spiral
    * hammer
    * roughness, circle_rough, ring_rough, fresnel_lens_rough,
"""

import datetime

import matplotlib.figure as mpfig
import matplotlib.image as mpimg
from numpy import (angle, arctan, arctan2, cos, exp, linspace, meshgrid, ones,
                   ones_like, pi, shape, sin, sqrt, zeros, zeros_like)
from PIL import Image
from scipy.signal import fftconvolve

from diffractio import degrees, np, plt, sp, um
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.utils_math import fft_convolution2d, nearest, nearest2
from diffractio.utils_optics import roughness_2D


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

    def set_amplitude(self, q=1, positivo=0, amp_min=0, amp_max=1):
        """makes that the mask has only maplitude

        Parameters:
            q (int): 0 - amplitude as it is and phase is removed. 1 - take phase and convert to amplitude

            positivo (int): 0 - value may be positivo or negative. 1 - value is only positive
        """

        amplitude = np.abs(self.u)
        phase = angle(self.u)

        if q == 0:
            if positivo == 0:
                self.u = amp_min + (
                    amp_max - amp_min) * amplitude * np.sign(phase)
            if positivo == 1:
                self.u = amp_min + (amp_max - amp_min) * amplitude
        else:
            if positivo == 0:
                self.u = amp_min + (amp_max - amp_min) * phase
            if positivo == 1:
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

    def save_mask(self, filename="", info=""):
        """Create a mask in a file, for example, ablation or litography engraver

        Parameters:
            filename (str): file name
            info (str): info of the mask

        Returns:
            float: area (in um**2)
        """

        # creo nombre para txt
        name = filename.split(".")
        nombreTxt = name[0] + ".txt"

        # creo image
        plt.figure()
        plt.imsave(filename, self.u, cmap='gray', dpi=300, origin='lower')
        plt.close()

        # creo txt con data importantes
        ofile = open(nombreTxt, "w")
        ofile.write("nombre de archivo %s\n" % filename)
        ofile.write("fecha: {}\n".format(datetime.date.today()))
        if info is not None:
            ofile.write("\ninfo:\n")
            ofile.write(info)
        ofile.write("\n\n")
        ofile.write("length de la mask: %i x %i\n" % (len(self.x),
                                                      len(self.y)))
        ofile.write("x0 = %f *um, x1 = %f *um, Deltax = %f *um\n" %
                    (self.x.min(), self.x[-1], self.x[1] - self.x[0]))
        ofile.write("y0 = %f *um, y1 = %f *um, Deltay = %f *um\n" %
                    (self.y.min(), self.y[-1], self.y[1] - self.y[0]))
        ofile.write("\nlongitud de onda = %f *um" % self.wavelength)
        ofile.close()

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
        filter.circle(
            r0=(0 * um, 0 * um), radius=(radius, radius), angle=0 * degrees)

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
            radius (float, float): size of mask
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

    def insert_array_masks(self, t1, space, margin=0, angle=0 * degrees):
        """Generates a matrix of shapes given in t1.

        Parameters:
            t1 (Scalar_mask_XY): Mask of the desired figure to be drawn
            space (float, float): spaces between figures.
            margin (float, float): extra space outside the mask
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

    def slit(self, x0, size, angle=0 * degrees):
        """Slit: 1 inside, 0 outside

        Parameters:
            x0 (float): center of slit
            size (float): size of slit
            angle (float): angle of rotation in radians
        """
        # Definicion de la slit
        xmin = x0 - size / 2
        xmax = x0 + size / 2

        # Rotacion de la slit
        Xrot, Yrot = self.__rotate__(angle)

        # Definicion de la transmitancia
        u = zeros(shape(self.X))
        ix = (Xrot < xmax) & (Xrot > xmin)
        u[ix] = 1
        self.u = u

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
            size (float, float): size of slit
            angle (float): angle of rotation in radians

        Example:

            m.square(r0=(0 * um, 0 * um), size=(250 * um, 120 * um), angle=0 * degrees)
        """

        # si solamente un numero, posiciones y radius son los mismos para ambos

        if len(size) == 1:
            size = (size, size)

        x0, y0 = r0
        # Tamano
        sizex, sizey = size

        # Definicion del square/rectangle
        xmin = x0 - sizex / 2
        xmax = x0 + sizex / 2
        ymin = y0 - sizey / 2
        ymax = y0 + sizey / 2

        # Rotacion del square/rectangle
        Xrot, Yrot = self.__rotate__(angle)

        # Transmitancia de los points interiores
        u = zeros(shape(self.X))
        ipasa = (Xrot < xmax) & (Xrot > xmin) & (Yrot < ymax) & (Yrot > ymin)
        u[ipasa] = 1
        self.u = u

    def one_level(self, level=0):
        """Sets one level for all the image.

        Parameters:
            level (float): value
        """
        self.u = level * ones(self.X.shape)

    def two_levels(self, level1=0, level2=1, xcorte=0, angle=0):
        """Divides the field in two levels

        Parameters:
            level1 (float): value of first level
            level2 (float): value of second level
            xcorte (float): position of division
            angle (float): angle of rotation in radians
        """
        Xrot, Yrot = self.__rotate__(angle)
        self.u = level1 * ones(self.X.shape)
        self.u[Xrot > xcorte] = level2

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
            radius (float, float): radius of circle/ellipse
            angle (float): angle of rotation in radians

        Example:

            circle(r0=(0 * um, 0 * um), radius=(250 * um, 125 * um), angle=0 * degrees)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos
        x0, y0 = r0

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle)

        # Definicion de la transmitancia
        u = zeros(shape(self.X))
        ipasa = (Xrot - x0)**2 / radiusx**2 + (Yrot - y0)**2 / radiusy**2 < 1
        u[ipasa] = 1
        self.u = u

    def super_gauss(self, r0, radius, potencia=2, angle=0 * degrees):
        """Supergauss mask.

        Parameters:
            r0 (float, float): center of circle
            radius (float, float): radius of circle
            potencia (float): value of exponential
            angle (float): angle of rotation in radians

        Example:

            super_gauss(r0=(0 * um, 0 * um), radius=(250 * um, 125 * um), angle=0 * degrees, potencia=2)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos

        if isinstance(radius, (float, int, complex)):
            radiusx, radiusy = (radius, radius)
        else:
            radiusx, radiusy = radius

        # Radios mayor y menor
        x0, y0 = r0

        # Rotacion del circula/elipse
        Xrot, Yrot = self.__rotate__(angle)
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

        Xrot, Yrot = self.__rotate__(angle)
        F = sqrt(Xrot**2 / R1**2 + Yrot**2 / R2**2 - s**2 * Xrot**2 * Yrot**2 /
                 (R1**2 * R2**2))

        Z1 = F < 1
        Z = Z1 * t1.u

        self.u = Z

    def ring(self, r0, radius1, radius2, angle=0 * degrees):
        """ Ring

        Parameters:
            r0 (float, float): center of ring
            radius1 (float, float): inner radius
            radius2 (float, float): outer radius
            angle (float): angle of rotation in radians
        """

        # si solamente un numero, posiciones y radius son los mismos para ambos
        # Definicion del origen y radius del ring

        if len(radius1) == 1:
            radius1 = (radius1, radius1)
        if len(radius2) == 1:
            radius2 = (radius2, radius2)

        ring1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        ring1.circle(r0, radius1, angle)
        ring2.circle(r0, radius2, angle)

        # Al restar ring2.u-ring1.u se logra la transmitancia en el interior
        self.u = ring2.u - ring1.u

    def cross(self, r0, size, angle=0 * degrees):
        """ Cross

        Parameters:
            r0 (float, float): center of cross
            size (float, float): length, width of cross
            angle (float): angle of rotation in radians
        """
        # Definicion del origen y length de la cross

        if len(size) == 1:
            size = (size[0], size[0])

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

    def prism(self, r0, index, angle_wedge_x, angle_wedge_y,
              angle=0 * degrees):
        """prism with angles angle_wedge_x, angle_wedge_y

        Parameters:
            r0 (float, float): center wedge
            index (float): refraction index
            angle_wedge_x (float): angle of wedge in x direction
            angle_wedge_y (float): angle of wedge in y direction
            angle (float): angle of rotation in radians

        """
        # Vector de onda
        k = 2 * pi / self.wavelength
        x0, y0 = r0
        Xrot, Yrot = self.__rotate__(angle)

        self.u = exp(1j * k * (index - 1) * (
            (Xrot - x0) * sin(angle_wedge_x)) +
                     (Yrot - y0) * sin(angle_wedge_y))

    def lens(self, r0, radius, focal, angle=0 * degrees, mask=True):
        """Transparent lens

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            focal (float, float): focal length of lens
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
        Xrot, Yrot = self.__rotate__(angle)

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
            radius (float, float): radius of lens mask
            focal (float, float): focal length of lens
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
        Xrot, Yrot = self.__rotate__(angle, position=r0)

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

    def lens_billet(self,
                    r0,
                    radius,
                    focal,
                    radius_agujero,
                    angle=0 * degrees,
                    mask=True):
        """Billet lens, that is a lens with a hole in the center

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float, float): radius of lens mask
            focal (float, float): focal length of lens
            radius_agujero (float): radius of hole
            angle (float): angle of axis in radians
            mask (bool): if True, mask with size radius
        """
        x0, y0 = r0
        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t1.lens(r0, radius, focal, angle, mask)
        field = t1.u
        ipasa = (t1.X - x0)**2 / radius_agujero**2 + (
            t1.Y - y0)**2 / radius_agujero**2 < 1
        field[ipasa] = 1
        self.u = field

    def axicon(self, r0, radius, height, n):
        """Axicon, that is a lens with a hole in the center

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            radius (float): radius of lens mask
            height (float): height of axicon
            n (float): refraction index

        Example:
            axicon(r0=(0 * um, 0 * um), radius=200 * um, height=5 * um,  n=1.5)
        """
        # Vector de onda
        k = 2 * pi / self.wavelength
        x0, y0 = r0

        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)

        # Altura desde la base a la surface
        h = -2 * height / radius * r + 2 * height
        # No existencia de heights negativas
        iremove = h < 0
        h[iremove] = 0

        # Region de transmitancia
        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * exp(1.j * k * (n - 1) * h)

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

    def radial_grating(self, r0, period, phase, radius, binaria=True):
        """Radial grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            binaria (bool): if True binary else, scaled

        Example:
            radial_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, radius=400 * um, binaria=True)
        """

        # Vector de onda
        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        # hago un seno y luego binarizo
        t = 0.5 * (1 + sin(2 * pi * (r - phase) / period))
        if binaria is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1
        # Region de transmitancia
        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1
        self.u = u * t

    def angular_grating(self, r0, period, phase, radius, binaria=True):
        """Angular grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            binaria (bool): if True binary else, scaled

        Example:
            angular_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, radius=400 * um, binaria=True)
        """
        # si solamente un numero, posiciones y radius son los mismos para ambos

        x0, y0 = r0
        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = (1 + sin(2 * pi * (theta - phase) / period)) / 2
        if binaria is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def hyperbolic_grating(self,
                           r0=(0 * um, 0 * um),
                           period=20 * degrees,
                           phase=0 * degrees,
                           radius=200 * um,
                           binaria=True,
                           angle=0 * degrees):
        """Hyperbolic grating.

        Parameters:
            r0 (float, float): (x0,y0) - center of lens
            period (float): period of the grating
            phase (float): initial phase
            radius (float): radius of the grating (masked)
            binaria (bool): if True binary else, scaled
            angle (float): angle of the grating in radians

        Example:
            hyperbolic_grating(r0=(0 * um, 0 * um), period=20 * um, phase=0 * um, sfradius=400 * um, binaria=True)
        """

        x0, y0 = r0
        # distance de la generatriz al eje del cono

        # rotation de la lens
        Xrot, Yrot = self.__rotate__(angle)

        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        x_posiciones = sqrt(np.abs((Xrot - x0)**2 - (Yrot - y0)**2))
        # Region de transmitancia
        t = (1 + sin(2 * pi * x_posiciones / period)) / 2
        if binaria is True:
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

        th1.square(
            r0=(xmin, ymin), size=(hammer_width, hammer_width), angle=angle)
        th2.square(
            r0=(xmin, ymax), size=(hammer_width, hammer_width), angle=angle)
        th3.square(
            r0=(xmax, ymin), size=(hammer_width, hammer_width), angle=angle)
        th4.square(
            r0=(xmax, ymax), size=(hammer_width, hammer_width), angle=angle)
        # La superposicion de ambas da lugar a la cross
        t3 = t1.u + th1.u + th2.u + th3.u + th4.u
        t3[t3 > 0] = 1
        self.u = t3

    def archimedes_spiral(self, r0, period, phase, p, radius, binaria):
        """Archimedes spiral

        Parameters:
            r0 (float, float): (x0,y0) - center of archimedes_spiral
            period (float): period of spiral
            phase (float): initial phase of spiral
            p (int): power of spiral
            radius (float): radius of the mask
            binaria (bool): if True binary mask

        Example:
            archimedes_spiral(r0=(0 * um, 0 * um), period=20 * degrees, phase=0 * degrees, p=1, radius=200 * um, binaria=True)
        """

        x0, y0 = r0

        # distance de la generatriz al eje del cono
        r = sqrt((self.X - x0)**2 + (self.Y - y0)**2)
        theta = arctan((self.Y - y0) / (self.X - x0))
        # Region de transmitancia
        t = 0.5 * (1 + sin(2 * pi * np.sign(self.X) *
                           ((r / period)**p + (theta - phase) / (2 * pi))))
        if binaria is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def laguerre_gauss_spiral(self, r0, kind, l, w0, z):
        """laguerre_gauss spiral

        Parameters:
            r0 (float, float): (x0,y0) - center of laguerre_gauss_spiral
            kind (str): 'amplitude' or 'phase'
            l (int): power of spiral
            w0 (float): width of spiral
            z (float): propagation distance

        Example:
            laguerre_gauss_spiral(r0=(0 * um, 0 * um), kind='amplitude', l=1, w0=625 * um, z=0.01 * um)
        """

        u_ilum = Scalar_source_XY(
            x=self.x, y=self.y, wavelength=self.wavelength)
        # Haz de Laguerre
        u_ilum.laguerre_beam(p=0, l=l, r0=r0, w0=w0, z=z)

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

        Xrot, Yrot = self.__rotate__(angle)

        THETA = arctan2(Xrot, Yrot)

        self.u = exp(
            1.j * alpha * cos(l * THETA - 2 * pi / period * (Xrot - r0[0])))

        phase = np.angle(self.u)

        phase[phase < 0] = 0
        phase[phase > 0] = 1

        if kind == 'amplitude':
            self.u = phase
        elif kind == 'phase':
            self.u = exp(1.j * pi * phase)

    def sine_grating(self,
                     period,
                     amp_min=0,
                     amp_max=1,
                     x0=0 * um,
                     angle=0 * degrees):
        """Sinusoidal grating:  self.u = amp_min + (amp_max - amp_min) * (1 + cos(2 * pi * (Xrot - phase) / period)) / 2

        Parameters:
            period (float): period of the grating
            amp_min (float): minimum amplitude
            amp_max (float): maximum amplitud
            x0 (float): phase shift
            angle (float): angle of the grating in radians

        Example:
             sine_grating(period=40 * um, amp_min=0, amp_max=1, x0=0 * um, angle=0 * degrees)
        """
        Xrot, Yrot = self.__rotate__(angle)

        # Definicion de la sinusoidal
        self.u = amp_min + (amp_max - amp_min) * (
            1 + cos(2 * pi * (Xrot - x0) / period)) / 2

    def sine_edge_grating(self,
                          r0=(0 * um, 0 * um),
                          period=20 * degrees,
                          lp=10 * um,
                          ap=2 * um,
                          phase=0 * degrees,
                          radius=200 * um,
                          binaria=True):
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
        if binaria is True:
            i0 = t <= 0.5
            t[i0] = 0
            i1 = t > 0.5
            t[i1] = 1

        u = zeros(shape(self.X))
        ipasa = r < radius
        u[ipasa] = 1

        self.u = u * t

    def ronchi_grating(self, period, fill_factor=0.5, x0=0 * um, angle=0):
        """Amplitude binary grating with fill factor: self.u = amp_min + (amp_max - amp_min) * (1 + cos(2 * pi * (Xrot - phase) / period)) / 2

        Parameters:
            period (float): period of the grating
            fill_factor (float): fill_factor
            x0 (float):  phase shift
            angle (float): angle of the grating in radians

        Notes:
            Ronchi grating when fill_factor = 0.5.

            It is obtained from a sinusoidal, instead as a sum of slits, for speed.

            The equation to determine the position y0 is: y0=cos(pi*fill_factor)

        Example:
            ronchi_grating(period=40*um, fill_factor=0.5, x0=0 * um, angle=0)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        y0 = cos(pi * fill_factor)

        t.sine_grating(
            period=period, amp_min=-1, amp_max=1, x0=x0, angle=angle)
        t.u[t.u > y0] = 1
        t.u[t.u <= y0] = 0
        self.u = t.u

    def binary_grating(self,
                       period,
                       amin=0,
                       amax=1,
                       phase=0 * degrees,
                       x0=0,
                       fill_factor=0.5,
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
            binary_grating(period=40 * um, amin=0, amax=1, phase=0 * degrees, x0=0, fill_factor=0.5, angle=0 * degrees)
        """
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t.ronchi_grating(period, fill_factor, x0, angle)
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
        Xrot, Yrot = self.__rotate__(angle)

        # Calculo de la pendiente
        pendiente = height / period
        # Calculo de la height
        h = (Xrot - x0) * pendiente

        # Calculo del a phase
        phase = k * (index - 1) * h
        # Definicion del origen
        phase = phase - phase.min()
        # Normalizacion entre 0 y 2pi
        phase = np.remainder(phase, 2 * pi)
        self.u = exp(1j * phase)

    def grating_2D(self,
                   period=40. * um,
                   amin=0,
                   amax=1.,
                   phase=0. * pi / 2,
                   x0=0,
                   fill_factor=0.75,
                   angle=0.0 * degrees):
        """2D binary grating

         Parameters:
            period (float): period of the grating
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            x0 (float):  phase shift
            fill_factor (float): fill_factor
            angle (float): angle of the grating in radians

        Example:
            grating_2D(period=40. * um, amin=0, amax=1., phase=0. * pi / 2, x0=0, fill_factor=0.75, angle=0.0 * degrees)
        """
        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        # Red horizontal
        t1.binary_grating(period, amin, amax, phase, x0, fill_factor, angle)
        # Red vertical
        t2.binary_grating(period, amin, amax, phase, x0, fill_factor,
                          angle + 90. * degrees)
        # Red binaria
        self.u = t1.u * t2.u

    def grating_2D_chess(self,
                         period=40 * um,
                         amin=0,
                         amax=1,
                         phase=0 * pi / 2,
                         x0=0,
                         fill_factor=0.75,
                         angle=0 * degrees):
        """2D binary grating as chess

         Parameters:
            period (float): period of the grating
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): max phase shift in phase gratings
            x0 (float):  phase shift
            fill_factor (float): fill_factor
            angle (float): angle of the grating in radians

        Example:
            grating_2D_chess(period=40. * um, amin=0, amax=1., phase=0. * pi / 2, x0=0, fill_factor=0.75, angle=0.0 * degrees)
        """

        t1 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t2 = Scalar_mask_XY(self.x, self.y, self.wavelength)
        t1.binary_grating(period, amin, amax, phase, x0, fill_factor, angle)
        t2.binary_grating(period, amin, amax, phase, x0, fill_factor,
                          angle + 90. * degrees)
        u = np.logical_xor(t1.u, t2.u)
        self.u = u.astype(float)

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
        ipasa = (Xrot - x0)**2 + (Yrot - y0)**2 - (
            radius + sigma * random_part)**2 < 0
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
            ring.ring_rough(
                r0,
                inner_radius,
                outer_radius,
                angle=angle,
                sigma=sigma,
                correlation_length=correlation_length)
            lens.u = lens.u + ring.u
        self.u = lens.u

    def super_ellipse(self,
                      r0=(0 * um, 0 * um),
                      radius=(50 * um, 50 * um),
                      angle=0 * degrees,
                      n=(2, 2)):
        """Super_ellipse. |(Xrot - x0) / radiusx|^n1 + |(Yrot - y0) / radiusy|=n2

        Parameters:
            r0 (float, float): center of super_ellipse
            radius (float, float): radius of the super_ellipse
            angle (float): angle of rotation in radians
            n (float, float) =  degrees of freedom of the next equation, n = (n1, n2)

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
        Xrot, Yrot = self.__rotate__(angle)

        # Definition of transmittance
        u = np.zeros_like(self.X)
        ipasa = np.abs((Xrot - x0) / radiusx)**nx + \
            np.abs((Yrot - y0) / radiusy)**ny < 1
        u[ipasa] = 1
        self.u = u

    def elliptical_phase(self, f1, f2, angle):

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

        Xrot, Yrot = self.__rotate__(angle)

        u = np.zeros_like(self.X)
        X_sin1 = x0 + size / 2 + amplitude1 * np.sin(
            2 * np.pi * Yrot / period1)
        X_sin2 = x0 - size / 2 + amplitude2 * np.sin(
            2 * np.pi * Yrot / period2 + phase)
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
        Xrot, Yrot = self.__rotate__(angle)

        u = np.zeros_like(self.X)
        Y1 = slope_x * np.abs(Xrot - x0) + y0
        Y2 = slope_y * np.abs(Xrot - x0) + y0

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

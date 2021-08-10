# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_mask_X class for definingn masks. Its parent is Scalar_field_X.

The main atributes are:
    * self.u - field
    * self.x - x positions of the field
    * self.wavelength - wavelength of the incident field. The field is monocromatic

*Class for unidimensional scalar masks*

*Functions*
    * mask_from_function, mask_from_array
    * slit, double slit
    * two_levels, gray_scale
    * prism, biprism_fresnel, biprism_fresnel_nh
    * lens, aspheric, fresnel_lens
    * roughness, dust, dust_different_sizes
    * sine_grating, ronchi_grating, binary_grating, blazed_grating
    * chriped_grating, chirped_grating_p, chirped_grating_q
    * I0s
"""

from scipy.interpolate import interp1d

from . import degrees, np, plt, um
from .scalar_fields_X import Scalar_field_X
from .utils_math import cut_function, fft_convolution1d, nearest, nearest2
from .utils_optics import roughness_1D


class Scalar_mask_X(Scalar_field_X):
    """Class for unidimensional scalar masks.

    Parameters:
        x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        wavelength (float): wavelength of the incident field
        n_background (float): refraction index of background
        info (str): String with info about the simulation

    Attributes:
        self.x (numpy.array): linear array with equidistant positions.
            The number of data is preferibly :math:`2^n` .
        self.wavelength (float): wavelength of the incident field.
        self.u (numpy.array): equal size than x. complex field
        self.quality (float): quality of RS algorithm
        self.info (str): description of data
        self.type (str): Class of the field
        self.date (str): date when performed
    """

    def __init__(self, x=None, wavelength=None, n_background=1, info=""):
        """equal than Scalar_field_X"""
        super(self.__class__, self).__init__(x, wavelength, n_background, info)
        self.type = 'Scalar_mask_X'

    def filter(self, mask, new_field=True, binarize=False, normalize=False):
        """Widens a field using a mask.

        Parameters:
            mask (diffractio.Scalar_mask_X): filter
            new_field (bool): If True, develope new Field
            binarize (bool, float): If False nothing, else binarize in level
            normalize (bool): If True divides the mask by sum:
        """

        f1 = np.abs(mask.u)

        if normalize is True:
            f1 = f1 / f1.sum()

        covolved_image = fft_convolution1d(f1, np.abs(self.u))
        if binarize is not False:
            covolved_image[covolved_image > binarize] = 1
            covolved_image[covolved_image <= binarize] = 0

        if new_field is True:
            new = Scalar_field_X(self.x, self.wavelength)
            new.u = covolved_image
            return new
        else:
            self.u = covolved_image

    def mask_from_function(self,
                           index=1.5,
                           f1=0,
                           f2=0,
                           v_globals={},
                           mask=True,
                           x0=0 * um,
                           radius=100 * um):
        """Phase mask defined between two surfaces :math:`f_1` and :math:`f_2`: :math:`h(x,y)=f_2(x,y)-f_1(x,y)`, :math:`t(x)=mask(x)e^{i\,k\,(n-1)(f_{2}-f_{1})}`

        Parameters:
            index (float): refraction index of the mask
            f1 (str): first surface
            f2 (str): second surface
            v_globals (dict): variable definitions
            mask (bool): True if a mask is defined to block areas
            x0 (float): center of mask
            radius (float): radius of the mask
        """

        k = 2 * np.pi / self.wavelength

        if mask is True:
            amplitude = Scalar_mask_X(self.x, self.wavelength)
            amplitude.slit(x0, radius)
            t = amplitude.u
        else:
            t = np.ones_like(self.X)

        v_locals = {'self': self, 'np': np, 'degrees': degrees}

        F2 = eval(f2, v_globals, v_locals)
        F1 = eval(f1, v_globals, v_locals)
        self.u = t * np.exp(1.j * k * (index - 1) * (F2 - F1))
        self.u[t == 0] = 0

    def mask_from_array(self,
                        index=1.5,
                        array1=None,
                        array2=None,
                        interp_kind='quadratic',
                        mask=True,
                        radius=100 * um,
                        x0=0 * um):
        """Phase mask defined between two surfaces defined by arrays: array1 and array2, :math:`t(x)=mask(x)e^{i\,k\,(n-1)(array2(x,z)-array1(x,z))}`

        Parameters:
            index (float): refraction index of the mask
            array1 (numpy.array): array of data (x,z) for the first surface
            array2 (numpy.array): array of data (x,z) for the second surface
            interp_kind (str):  'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            mask (bool): True if a mask is defined to block areas
            x0 (float): center of mask
            radius (float): radius of the mask
        """

        k = 2 * np.pi / self.wavelength

        if mask is True:
            amplitude = Scalar_mask_X(self.x, self.wavelength)
            amplitude.slit(x0, radius)
            t = amplitude.u
        else:
            t = np.ones_like(self.x)

        f1_interp = interp1d(
            array1[:, 0],
            array1[:, 1],
            kind=interp_kind,
            bounds_error=False,
            fill_value=0)
        f2_interp = interp1d(
            array2[:, 0],
            array2[:, 1],
            kind=interp_kind,
            bounds_error=False,
            fill_value=0)  # interpolates all the range

        F1 = f1_interp(self.x)
        F2 = f2_interp(self.x)

        self.u = t * np.exp(1.j * k * (index - 1) * (F2 - F1))
        self.u[t == 0] = 0

    def dots(self, x0):
        """Generates 1 or several point masks at positions x0

        Parameters:
            x0 float or np.array: x point or points where mask is 1.
        """
        u = np.zeros_like(self.x)

        if type(x0) in (int, float):
            i_x0, _, _ = nearest(self.x, x0)
            u[i_x0] = 1
        else:
            i_x0s, _, _ = nearest2(self.x, x0)
            for i_x0 in i_x0s:
                u[i_x0] = 1

        self.u = u

    def slit(self, x0=0, size=100 * um):
        """Slit: 1 inside, 0 outside

        Parameters:
            x0 (float): center of slit
            size (float): size of slit
        """

        # Definicion de la slit
        xmin = x0 - size / 2
        xmax = x0 + size / 2

        # Definicion de la transmitancia
        u = np.zeros_like(self.x)
        ix = (self.x < xmax) & (self.x > xmin)
        u[ix] = 1
        self.u = u

        return u

    def double_slit(self, x0, size, separation):
        """double slit: 1 inside, 0 outside

        Parameters:
            x0 (float): center of slit
            size (float): size of slit
            separation (float): separation between slit centers
        """

        slit1 = Scalar_mask_X(self.x, self.wavelength)
        slit2 = Scalar_mask_X(self.x, self.wavelength)

        # Definicion de las dos slits
        slit1.slit(x0=x0 - separation / 2, size=size)
        slit2.slit(x0=x0 + separation / 2, size=size)
        self.u = slit1.u + slit2.u

        return self.u

    def two_levels(self, level1=0, level2=1, x_edge=0):
        """Divides the image in two levels.

        Parameters:
            level1 (float): value of level1
            level2 (float): value of level2
            x_edge (float): position of separation of levels
        """

        # Definicion de un primer level de unos
        self.u = level1 * np.ones(self.x.shape)
        # Aquellos points cuyo mayor sea mayor que el value de corte, adquieren
        # el value del level2
        self.u[self.x > x_edge] = level2

    def gray_scale(self, num_levels, levelMin=0, levelMax=1):
        """Divides the mask in n, vertical levels.

        Parameters:
            num_levels (int): number of levels
            levelMin (float): minimum value of levels
            levelMax (float): maximum value of levels
        """
        t = np.zeros(self.x.shape, dtype=float)

        xpos = np.linspace(self.x[0], self.x[-1], num_levels + 1)
        height_levels = np.linspace(levelMin, levelMax, num_levels)
        ipos, _, _ = nearest2(self.x, xpos)
        ipos[-1] = len(self.x)

        for i, h_level in enumerate(height_levels):
            t[ipos[i]:ipos[i + 1]] = h_level

        self.u = t

    def prism(self, x0, n, anglex):
        """Prism.

        Parameters:
            x0 (float): vertex of prism
            n (float): refraction_index
            anglex (float): angle of prism

        TODO:
            I think it is wrong: include refraction index
        """

        k = 2 * np.pi / self.wavelength

        h = (self.x - x0) * np.sin(anglex)
        self.u = np.exp(1j * k * (n - 1) * h)

        h = h - h.min()
        return h

    def biprism_fresnel(self, angle, x0, radius, mask):
        """Fresnel biprism.

        Parameters:
            angle (float): angle of the fresnel biprism
            x0 (float): central position of fresnel biprism
            radius (float): radius of the fresnel biprism, if mask is True
            mask (bool): if True radius is applied
        """

        k = 2 * np.pi / self.wavelength

        # distance from generatriz to cone axis
        xp = self.x > x0
        xn = self.x <= x0

        # heigth
        h = np.zeros_like(self.x)
        h[xp] = -np.sin(angle) * (self.x[xp] - x0)
        h[xn] = np.sin(angle) * (self.x[xn] - x0)

        u = np.exp(1.j * (k * h + np.pi))

        t = np.ones_like(self.x)

        if mask is True:
            ipasa = np.abs(self.x - x0) > radius
            t[ipasa] = 0.
            remove_phase_out = np.angle(u)
            remove_phase_out[ipasa] = 0
            u = np.abs(u) * np.exp(1j * remove_phase_out)

        self.u = u * t

    def biprism_fresnel_nh(self,
                           x0,
                           width,
                           height,
                           n):
        """Fresnel biprism, uses height and refraction index.

        Parameters:
            x0 (float): vertex of biprism
            width (float): size of biprism
            height (float): height of biprism
            n (float): refraction_index
        """

        k = 2 * np.pi / self.wavelength

        xp = self.x > 0
        xn = self.x <= 0

        # heigth
        h = np.zeros_like(self.x)
        h[xp] = -2 * height / width * (self.x[xp] - x0) + 2 * height
        h[xn] = 2 * height / width * (self.x[xn] - x0) + 2 * height

        # No existencia de heights negativas
        iremove = h < 0
        h[iremove] = 0

        # Region de transmitancia
        u = np.zeros_like(self.x)
        ipasa = np.abs(self.x - x0) < width
        u[ipasa] = 1

        self.u = u * np.exp(1.j * k * (n - 1) * h)
        return h

    def lens(self, x0, focal, radius, mask=True):
        """Transparent lens.

        Parameters:
            x0 (float): center of lens
            focal (float): focal length of lens
            mask (bool): if True, mask with size radius
            radius (float): radius of lens mask
        """

        k = 2 * np.pi / self.wavelength

        # Definicion de la amplitude y la phase
        if mask is True:
            # amplitude = Scalar_mask_X(self.x, self.wavelength)
            # amplitude.slit(x0, 2 * radius)
            # t = amplitude.u
            t = np.zeros_like(self.x, dtype=np.int)
            ix = (self.x < x0 + radius) & (self.x > x0 - radius)
            t[ix] = 1
        else:
            t = np.ones_like(self.x, dtype=np.int)

        h = (self.x - x0)**2 / (2 * focal)
        self.u = t * np.exp(-1.j * k * h)
        self.u[t == 0] = 0

        h = h - h.min()
        h = h / h.max()
        return t * h

    def aspheric(self, x0, c, k, a, n0, n1, radius, mask=True):
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

        s2 = (self.x - x0)**2
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
                     x0,
                     focal,
                     kind='amplitude',
                     binary=True,
                     phase=np.pi,
                     mask=True,
                     radius=100 * um):
        """Fresnel lens. Amplitude phase, continuous or binary.

        Parameters:
            x0 (float): center of lens
            focal (float): focal length of lens
            kind (str): 'amplitude' or phase
            binary (bool): binary or profile
            phase (float): if kind=='phase' -> maximum phase
            mask (bool): if True, mask with size radius
            radius (float): radius of lens mask

        Returns:
            h (np.array): normalized heights [0,1] of lens
        """
        # Vector de onda
        k = 2 * np.pi / self.wavelength

        # Definicion de la amplitude y la phase
        if mask is True:

            t1 = np.zeros_like(self.x)
            ix = (self.x < x0 + radius) & (self.x > x0 - radius)
            t1[ix] = 1
        else:
            t1 = np.ones_like(self.x)

        if kind == 'amplitude' and binary is True:
            u_fresnel = np.cos(k * ((self.x - x0)**2 / (2 * focal)))
            u_fresnel[u_fresnel > 0] = 1
            u_fresnel[u_fresnel <= 0] = 0

        elif kind == 'phase' and binary is True:
            u_fresnel = np.cos(k * ((self.x - x0)**2 / (2 * focal)))
            u_fresnel[u_fresnel > 0] = 1
            u_fresnel[u_fresnel <= 0] = 0
            u_fresnel = np.exp(1j * u_fresnel * phase)

        elif kind == 'phase' and binary is False:
            u_fresnel = np.exp(-1j * k * ((self.x - x0)**2 / (2 * focal)))
            fase = np.angle(u_fresnel) / (2 * np.pi)
            u_fresnel = np.exp(1j * fase * phase)

        elif kind == 'amplitude' and binary is False:
            u_fresnel = (1 + np.cos(k * ((self.x - x0)**2 / (2 * focal)))) / 2

        h = u_fresnel
        h = h - h.min()

        self.u = u_fresnel * t1
        h = t1 * h
        return h / h.max()

    def roughness(self, t=50 * um, s=1 * um):
        """Rough surface, phase

        According to movile average (Ogilvy p.224).
        very good in time for long arrays

        Parameters:
            t (float): correlation lens
            s (float): std of roughness

        Returns:
            numpy.array: topography maps in microns
        """

        h_corr = roughness_1D(self.x, t, s)
        k = 2 * np.pi / self.wavelength
        u = np.exp(-1.j * k * 2 * h_corr)
        u = u[0:len(self.x)]
        h_corr = h_corr[0:len(self.x)]
        self.u = u
        return h_corr

    def dust_different_sizes(self, percentaje, size, std=0):
        """Mask with dust particles of different sizes.

        Parameters:
            percentaje (float): percentaje of area afected by noise
            size (float): mean size of dust
            std (float): std for size of dust

        Returns:
            numpy.array: positions - positions of dust
            numpy.array: sizes - size of dust
            float: percentaje_real - real percentaje of dust
        """

        total_length = self.x[-1] - self.x[0]
        num_particles = int(percentaje * total_length / size)
        if percentaje > 0.5:
            num_particles = int(num_particles * (1 + np.sqrt(percentaje)))
        sizes = size + std * np.random.randn(num_particles)
        sizes[sizes < 0] = size
        positions = self.x[0] + total_length * np.random.rand(num_particles)

        dust = Scalar_mask_X(self.x, self.wavelength, 'dust')
        dust.u = np.ones_like(self.x)

        tmp = Scalar_mask_X(self.x, self.wavelength, 'dust')

        for i in range(num_particles):
            tmp.slit(x0=positions[i], size=sizes[i])
            dust.u = dust.u * (1 - tmp.u)

        # dust.u[dust.u > value] = value
        # dust.u = 1 - dust.u
        # dust.u[dust.u < 1] = value
        self.u = dust.u
        # can be used to increase dust_particles, when there is overlapping
        percentaje_real = 1 - self.u.sum() / len(self.x)

        return positions, sizes, percentaje_real

    def dust(self, percentaje, size=0):
        """ Mask with dust particles of equal sizes.

        Parameters:
            percentaje (float): percentaje of area afected by noise
            size (float): size of dust
            value (float): value included when there is noise

        Returns:
            numpy.array: positions - positions of dust
            numpy.array: sizes - size of dust
            float: percentaje_real - real percentaje of dust
        """

        total_length = self.x[-1] - self.x[0]
        dx = self.x[1] - self.x[0]
        i_center = int(len(self.x) / 2)
        num_particles = int(percentaje * total_length / size)
        if percentaje > 0.5:
            num_particles = int(num_particles * (1 + np.sqrt(percentaje)))
        # habria que quitar algo por los solapamientos
        positions = self.x[0] + total_length * np.random.rand(num_particles)

        dust = np.zeros_like(self.x)
        i_positions, _, _ = nearest2(self.x, positions)
        dust[i_positions] = 1

        filtro = np.zeros_like(self.x)
        num_pixels_2 = int(size / (2 * dx))
        filtro[i_center - num_pixels_2:i_center + num_pixels_2] = 1

        dust = fft_convolution1d(dust, filtro)
        dust[dust > 1] = 1

        self.u = 1 - dust

        percentaje_real = self.u.sum() / len(self.x)

        return positions, percentaje_real

    def sine_grating(self, x0, period, amp_min=0, amp_max=1):
        """Sinusoidal grating

        Parameters:
            x0 (float): shift of grating
            period (float): period of the grating
            amp_min (float): minimum amplitude
            amp_max (float): maximum amplitude
        """
        # Definicion de la sinusoidal
        self.u = amp_min + (amp_max - amp_min) * (
            1 + np.cos(2 * np.pi * (self.x - x0) / period)) / 2

        return self.u

    def ronchi_grating(self, x0, period, fill_factor=0.5):
        """Amplitude binary grating, fill-factor can be defined. It is obtained as a sine_grating that after is binarized. Fill factor is determined as  y0=cos(pi*fill_factor)

        Parameters:
            x0 (float): shift of the grating
            period (float): period of the grating
            fill_factor (float): (0,1) - fill factor of grating
        """

        t = Scalar_mask_X(self.x, self.wavelength)
        y0 = np.cos(np.pi * fill_factor)
        t.sine_grating(period=period, amp_min=-1, amp_max=1, x0=x0)
        t.u[t.u > y0] = 1
        t.u[t.u <= y0] = 0
        self.u = t.u
        return t.u

    def binary_grating(self,
                       x0,
                       period,
                       fill_factor,
                       amin,
                       amax,
                       phase):
        """binary grating amplitude and/or phase

        Parameters:
            x0 (float): shift of the grating
            period (float): period of the grating
            fill_factor (float): (0,1) - fill factor of grating
            amin (float): minimum amplitude
            amax (float): maximum amplitude
            phase (float): phase shift (radians)
        """

        t = Scalar_mask_X(self.x, self.wavelength)
        t.ronchi_grating(period=period, x0=x0, fill_factor=fill_factor)
        self.u = amin + (amax - amin) * t.u
        self.u = self.u * np.exp(1j * phase * t.u)
        return t.u

    def blazed_grating(self, x0, period, height, n, ):
        """Phase, blazed grating. The phase shift is determined by heigth and refraction index.

        Parameters:
            x0 (float): shift of the grating
            period (float): period of the grating
            height (float): height of grating
            n (float): refraction index of grating
        """

        k = 2 * np.pi / self.wavelength

        # Slope computation
        slope = height / period

        # Height computation
        h = (self.x - x0) * slope

        # Phase computation
        phase = k * (n - 1) * h

        # removing phase min
        phase = phase - phase.min()

        # normalization between 0 and 2pi
        phase = np.remainder(phase, 2 * np.pi)
        self.u = np.exp(1j * phase)
        return phase

    def chirped_grating_p(self,
                          kind,
                          p0,
                          p1,
                          amp_min,
                          amp_max,
                          phase_max,
                          delta_x=0,
                          x0=None,
                          length=0,
                          x_center=0):
        """Chirped grating with linear p(x) variation.

        Parameters:
            kind (str): 'amplitude', 'phase', 'amplitude_binary', 'phase_binary'
            p0 (float): initial period of the grating
            p1 (float): final period of the grating
            amp_min (float): minimum  transmittance
            amp_max (float): maximum transmittance
            phase_max (float): maximum modulation for phase gratings
            delta_x (float): x shifting for movement of grating
            x0 (float):  -
            length (float): length of the grating.  0: length is equal to size of x l=(x[-1]-x[0]),  <l: it can be shorter than l
            x_center (float): x-position of center of grating

        Returns:
            numpy.array: px
        """

        mask = Scalar_mask_X(self.x, self.wavelength)

        if length == 0 or length == self.x[-1] - self.x[0]:
            size = self.x[-1] - self.x[0]
            mask.u = np.ones_like(self.x)
            x0 = self.x[0] - delta_x
        elif length < self.x[-1] - self.x[0]:
            """
            es sencillo. se hace una red del tamaño l nueva (como si fuera 0)
            y luego se añade a la que hay
            """
            size = length
            x0 = self.x[0] - delta_x
            x1 = np.linspace(0, length, len(self.x))
            red1 = Scalar_mask_X(x1, self.wavelength)
            conds = {
                'kind': kind,
                'p0': p0,
                'p1': p1,
                'amp_min': amp_min,
                'amp_max': amp_max,
                'delta_x': delta_x,
                'phase_max': phase_max,
                'length': 0,
                'x_center': 0,
            }
            px = red1.chirped_grating_p(**conds)
            px = np.zeros_like(px, dtype=float)  # sale mal en este formato
            self.insert_mask(red1, x_center, kind_position='center')
            return px

        else:
            size = self.x[-1] - self.x[0]
            mask.u = np.ones_like(self.x)
            print("possible error in chriped_grating_q: length > x[-1]-x[0]")
            x0 = self.x[0] - delta_x

        pa = (p1 - p0) / size

        px = 2. * np.pi * np.log(p0 + pa * (self.x - x0)) / pa
        t = amp_min + (amp_max - amp_min) * (1 + np.cos(px)) / 2

        if kind == 'amplitude_binary' or kind == 'phase_binary':
            levels = [0, 1]
            corte = 0.5
            t_binaria = np.zeros_like(t, dtype='float')
            t_binaria[t <= corte] = levels[0]
            t_binaria[t > corte] = levels[1]
            t = t_binaria

        if kind == 'amplitude':
            self.u = t
        elif kind == 'phase':
            self.u = np.exp(1.j * phase_max * t)
            print(np.angle(self.u))
        elif kind == 'amplitude_binary':
            self.u = t_binaria
        elif kind == 'phase_binary':
            self.u = np.exp(1.j * phase_max * t_binaria)
        else:
            print("kind of chirped_grating_q not well defined")

        amplitud = self.u * mask.u
        fase = np.angle(self.u) * (1 - mask.u)
        self.u = amplitud * np.exp(1j * fase)
        px = (p0 + pa * (self.x - x0)) * mask.u

        return px, t

    def chirped_grating_q(self,
                          kind,
                          p0,
                          p1,
                          amp_min,
                          amp_max,
                          phase_max,
                          delta_x=0,
                          length=0,
                          x_center=0):
        """Chirped grating with linear q(x) variation. The transmitance is: t = np.cos(np.pi*q*(x-x0) + np.pi*q0*(x-x0))

        Parameters:
            kind (str): 'amplitude', 'phase', 'amplitude_binary', 'phase_binary'
            p0 (float): initial period of the grating
            p1 (float): final period of the grating
            amp_min (float): minimum  transmittance
            amp_max (float): maximum transmittance
            phase_max (float): maximum modulation for phase gratings
            delta_x (float): x shifting for movement of grating
            length (float): length of the grating,  0: length is equal to size of x l=(x[-1]-x[0]). <l: it can be shorter than l
            x_center (float): x-position of center of grating

        Returns:
            numpy.array: qx
        """

        mascara = Scalar_mask_X(self.x, self.wavelength)

        if length == 0 or length == self.x[-1] - self.x[0]:
            size = self.x[-1] - self.x[0]
            mascara.u = np.ones_like(self.x)
            x0 = self.x[0] - delta_x
        elif length < self.x[-1] - self.x[0]:
            size = length
            x0 = self.x[0] - delta_x
            x1 = np.linspace(0, length, len(self.x))
            red1 = Scalar_mask_X(x1, self.wavelength)
            conds = {
                'kind': kind,
                'p0': p0,
                'p1': p1,
                'amp_min': amp_min,
                'amp_max': amp_max,
                'delta_x': delta_x,
                'phase_max': phase_max,
                'length': 0,
                'x_center': 0,
            }
            qx = red1.chirped_grating_q(**conds)
            qx = np.zeros_like(qx, dtype=float)  # sale mal en este formato
            self.insert_mask(red1, x_center, kind_position='center')
            return qx
        else:
            size = self.x[-1] - self.x[0]
            mascara.u = np.ones_like(self.x)
            print("possible error in chriped_grating_q: length > x[-1]-x[0]")
            x0 = self.x[0] - delta_x

        q0 = 2 * np.pi / p0
        q1 = 2 * np.pi / p1

        qa = (q1 - q0) / size
        qx = q0 + 0.5 * qa * (self.x - x0)

        t = amp_min + (amp_max - amp_min) * (1 + np.cos(qx *
                                                        (self.x - x0))) / 2

        if kind == 'amplitude_binary' or kind == 'phase_binary':
            levels = [0, 1]
            corte = 0.5
            t_binaria = np.zeros_like(t, dtype='float')
            t_binaria[t <= corte] = levels[0]
            t_binaria[t > corte] = levels[1]
            t = t_binaria

        if kind == 'amplitude':
            self.u = t
        elif kind == 'phase':
            self.u = np.exp(1.j * phase_max * t)
            print(np.angle(self.u))
        elif kind == 'amplitude_binary':
            self.u = t_binaria
        elif kind == 'phase_binary':
            self.u = np.exp(1.j * phase_max * t_binaria)
        else:
            print("kind of chirped_grating_q not well defined")

        amplitud = self.u * mascara.u
        fase = np.angle(self.u) * (1 - mascara.u)
        self.u = amplitud * np.exp(1j * fase)
        qx = (q0 + qa * (self.x - x0)) * mascara.u

        return qx, t

    def chirped_grating(self,
                        kind,
                        p_x,
                        x0,
                        amp_min,
                        amp_max,
                        phase_max,
                        delta_x,
                        length=0):
        """General chirped grating with variation given by function p(x).

        Parameters:
            kind (str): 'amplitude', 'phase', 'amplitude_binary', 'phase_binary'
            p_x (str): function with variation of periods
            amp_min (float): minimum  transmittance
            amp_max (float): maximum transmittance
            phase_max (float): maximum modulation for phase gratings
            delta_x (float): x shifting for movement of grating
            length (float): length of the grating. 0: length is equal to size of x l=(x[-1]-x[0]).  <l: it can be shorter than l

        Returns:
            numpy.array: p(x)
        """

        if length == 0 or length == []:
            length = self.x[-1] - self.x[0]

        period = eval(p_x)
        q_x = 2 * np.pi / period
        t = amp_min + (amp_max - amp_min) * (
            1 + np.cos(q_x * (self.x - delta_x))) / 2
        # plt.figure()
        # plt.plot(t)

        if kind == 'amplitude_binary' or kind == 'phase_binary':
            levels = [0, 1]
            corte = 0.5
            t_binaria = np.zeros_like(t, dtype='float')
            t_binaria[t <= corte] = levels[0]
            t_binaria[t > corte] = levels[1]
            t = t_binaria

        if kind == 'amplitude':
            self.u = t
        elif kind == 'phase':
            self.u = np.exp(1.j * phase_max * t)
        elif kind == 'amplitude_binary':
            self.u = t_binaria
        elif kind == 'phase_binary':
            self.u = np.exp(1.j * phase_max * t_binaria)
        self.u = cut_function(self.x, self.u, length, '')

        return t

    def binary_code_positions(self, x_transitions, start='down', has_draw=True):
        """
        Genenerates a binary code, using the positions given in x_transitions

        Parameters:
            x_transitions (numpy.array): positions where transitions are placed
            start (str): How the binary code starts:'down' (starts in 0) or 'up' (starts in 1)
            has_draw (bool): If True, draws the code

        """

        x_transitions = np.sort(x_transitions)
        x_transitions = np.hstack((self.x[0], x_transitions, self.x[-1]))
        x_transitions = np.unique(x_transitions)

        t = np.ones_like(self.x)

        i_transitions, _, _ = nearest2(self.x, x_transitions)

        i_transitions = i_transitions.astype(int)

        for i in range(0, len(i_transitions) - 1, 2):
            i0 = i_transitions[i]
            i1 = i_transitions[i + 1]
            t[i0:i1] = 0

        if start == 'up':
            t = 1 - t

        self.u = t

        if has_draw:
            plt.figure(figsize=(18, 5))
            plt.plot(self.x, t)
            plt.plot(x_transitions, np.ones_like(x_transitions), 'ko')
            plt.xlim(self.x[0], self.x[-1])

        return t

    def binary_code(self,
                    x0=0 * um,
                    kind='standard',
                    code=[1, 1, 0, 0, 1, 0, 1],
                    bit_width=20 * um):
        """Binary code in form of 1's and 0's.

        Parameters:
            kind (str): there are serveral types of codes
                'standard' - normal
                'abs_fag' -  used in some abs encoders
            code (numpy.array): array with values of code
            bit_width (float): size of each data of code
            x0 (float): Initial position
        """

        if kind == 'abs_fag':
            i0_ones = np.ones_like(code)
            i0_zeros = np.zeros_like(code)
            code = np.vstack((i0_zeros, i0_ones, code, i0_ones)).reshape((-1, ),
                                                                         order='F')
            bit_width = bit_width / 4

        t = Scalar_mask_X(self.x, self.wavelength)
        t2 = Scalar_mask_X(self.x, self.wavelength)
        t.u = np.zeros(self.x.shape)

        for i0j, j in zip(code, list(range(len(code)))):
            t2.slit(x0 + (j + 0.5) * bit_width, bit_width)
            t.u = t.u + i0j * t2.u
        self.u[-1] = self.u[-2]
        self.u = t.u

        return t.u

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Common functions to classes """

from copy import deepcopy
from math import factorial

import numpy as np
import scipy.ndimage as ndimage
from numpy import angle, array, exp, linspace, ones_like, pi, zeros
from scipy.signal import fftconvolve

from . import mm


def distance(x1, x2):
    """Compute distance between two vectors.

    Parameters:
        x1 (numpy.array): vector 1
        x2 (numpy.array): vector 2

    Returns:
        (float): distance between vectors.
    """
    if len(x1) != len(x2):
        raise Exception('distance: arrays with different number of elements')
    else:
        return np.linalg.norm(x2 - x1)


def nearest(vector, number):
    """Computes the nearest element in vector to number.

    Parameters:
        vector (numpy.array): array with numbers
        number (float):  number to determine position

    Returns:
        (int): index - index of vector which is closest to number.
        (float): value  - value of vector[index].
        (float): distance - difference between number and chosen element.
    """
    indexes = np.abs(vector - number).argmin()
    values = vector.flat[indexes]
    distances = values - number
    return indexes, values, distances


def nearest2(vector, numbers):
    """Computes the nearest element in vector to numbers.

    Parameters:
        vector (numpy.array): array with numbers
        number (numpy.array):  numbers to determine position

    Returns:
        (numpy.array): index - indexes of vector which is closest to number.
        (numpy.array): value  - values of vector[indexes].
        (numpy.array): distance - difference between numbers and chosen elements.
    """

    indexes = np.abs(np.subtract.outer(vector, numbers)).argmin(0)
    values = vector[indexes]
    distances = values - numbers
    return indexes, values, distances


def find_extrema(array2D, x, y, kind='max', verbose=False):
    """In a 2D-array, formed by vectors x, and y, the maxima or minima are found

    Parameters:
        array2D (np. array 2D): 2D array with variable
        x (np.array 1D): 1D array with x axis
        y (np.array 1D): 1D array with y axis
        kind (str): 'min' or 'max': detects minima or maxima
        verbose (bool): If True prints data.

    Returns:
        indexes (int,int): indexes of the position
        xy_ext (float, float): position of maximum
        extrema (float): value of maximum
    """

    if kind == 'max':
        result = np.where(array2D == np.amax(array2D))
    elif kind == 'min':
        result = np.where(array2D == np.min(array2D))

    listOfCordinates = list(zip(result[1], result[0]))

    num_extrema = len(listOfCordinates)

    indexes = np.zeros((num_extrema, 2), dtype=np.integer)
    xy_ext = np.zeros((num_extrema, 2))
    extrema = np.zeros((num_extrema))

    for i, cord in enumerate(listOfCordinates):
        indexes[i, :] = cord[0], cord[1]
        xy_ext[i, 0] = x[cord[0]]
        xy_ext[i, 1] = y[cord[1]]
        extrema[i] = array2D[cord[1], cord[0]]

    if verbose is True:
        for cord in listOfCordinates:
            print(cord, x[cord[0]], y[cord[1]], array2D[cord[1], cord[0]])

    return indexes, xy_ext, extrema


def ndgrid(*args, **kwargs):
    """n-dimensional gridding like Matlab's NDGRID

    Parameters:
        The input *args are an arbitrary number of numerical sequences, e.g. lists, arrays, or tuples.
        The i-th dimension of the i-th output argument
        has copies of the i-th input argument.

    Example:

        >>> x, y, z = [0, 1], [2, 3, 4], [5, 6, 7, 8]

        >>> X, Y, Z = ndgrid(x, y, z)
        # unpacking the returned ndarray into X, Y, Z

        Each of X, Y, Z has shape [len(v) for v in x, y, z].

        >>> X.shape == Y.shape == Z.shape == (2, 3, 4)
        True

        >>> X
        array([[[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],
                   [[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]]])
        >>> Y
        array([[[2, 2, 2, 2],
                        [3, 3, 3, 3],
                        [4, 4, 4, 4]],
                   [[2, 2, 2, 2],
                        [3, 3, 3, 3],
                        [4, 4, 4, 4]]])
        >>> Z
        array([[[5, 6, 7, 8],
                        [5, 6, 7, 8],
                        [5, 6, 7, 8]],
                   [[5, 6, 7, 8],
                        [5, 6, 7, 8],
                        [5, 6, 7, 8]]])

        With an unpacked argument list:

        >>> V = [[0, 1], [2, 3, 4]]

        >>> ndgrid(*V) # an array of two arrays with shape (2, 3)
        array([[[0, 0, 0],
                        [1, 1, 1]],
                   [[2, 3, 4],
                        [2, 3, 4]]])

        For input vectors of different data kinds,
        same_dtype=False makes ndgrid()
        return a list of arrays with the respective dtype.
        >>> ndgrid([0, 1], [1.0, 1.1, 1.2], same_dtype=False)
        [array([[0, 0, 0], [1, 1, 1]]),
         array([[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]])]

        Default is to return a single array.

        >>> ndgrid([0, 1], [1.0, 1.1, 1.2])
        array([[[ 0. ,  0. ,  0. ], [ 1. ,  1. ,  1. ]],
                   [[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]]])
    """
    same_dtype = kwargs.get("same_dtype", True)
    V = [array(v) for v in args]  # ensure all input vectors are arrays
    shape = [len(v) for v in args]  # common shape of the outputs
    result = []
    for i, v in enumerate(V):
        # reshape v so it can broadcast to the common shape
        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        zero = zeros(shape, dtype=v.dtype)
        thisshape = ones_like(shape)
        thisshape[i] = shape[i]
        result.append(zero + v.reshape(thisshape))
    if same_dtype:
        return array(result)  # converts to a common dtype
    else:
        return result  # keeps separate dtype for each output


# def meshgrid2(*arrs):
#     arrs = tuple(reversed(arrs))  # edit
#     lens = map(len, arrs)
#     dim = len(arrs)
#
#     sz = 1
#     for s in lens:
#         sz *= s
#
#     ans = []
#     for i, arr in enumerate(arrs):
#         slc = [1] * dim
#         slc[i] = lens[i]
#         arr2 = np.asarray(arr).reshape(slc)
#         for j, sz in enumerate(lens):
#             if j != i:
#                 arr2 = arr2.repeat(sz, axis=j)
#         ans.append(arr2)
#
#     return tuple(ans)


def get_amplitude(u, sign=False):
    """Gets the amplitude of the field.

    Parameters:
        u (numpy.array): Field.
        sign (bool): If True, sign is kept, else, it is removed

    Returns:
        (numpy.array): numpy.array
    """

    amplitude = np.abs(u)

    if sign is True:
        phase = np.angle(u)
        amplitude = np.sign(phase) * amplitude

    return amplitude


def get_phase(u):
    """Gets the phase of the field.

    Parameters:
        u (numpy.array): Field.

    Returns:
        (numpy.array): numpy.array
    """
    phase = np.exp(1j * np.angle(u))
    return phase


def amplitude2phase(u):
    """Passes the amplitude of a complex field to phase. Previous phase is removed. :math:`u = A e^{i \phi}  -> e^(i abs(A))`

    Parameters:
        u (numpy.array, dtype=complex): complex field

    Returns:
        (numpy.array): only-phase complex vector.
    """

    amplitude = np.abs(u)
    u_phase = np.exp(1.j * amplitude)

    return u_phase


def phase2amplitude(u):
    """Passes the phase of a complex field to amplitude.

    Parameters:
        u (numpy.array, dtype=complex): complex field

    Returns:
        (numpy.array): amplitude without phase complex vector.
    """
    phase = np.angle(u)
    u_amplitud = phase

    return u_amplitud


def normalize(v, order=2):
    """Normalize vectors with different L norm (standard is 2).

    Parameters:
        v (numpy.array): vector to normalize
        order (int): order for norm

    Returns:
        (numpy.array): normalized vector.
    """

    norm = np.linalg.norm(v, ord=order)
    if norm == 0:
        raise ValueError('normalize: norm = 0.')
    return v / norm


#
# def normalize_field(u, kind='intensity'):
#     """Normalize the field.
#
#     Parameters:
#         u (np.array): field
#         kind (str): normalization parameter -'intensity' 'area'
#
#     Returns:
#         normalized value
#
#
#     """
#
#     if kind == 'intensity':
#         intensity = np.abs(u**2)
#         maximum = sqrt(intensity.max())
#         u = u / maximum
#     if kind == 'area':
#         intensity = np.abs(u**2)
#         maximum = intensity.sum()
#         u = u / maximum
#
#         return u


def binarize(vector, min_value=0, max_value=1):
    """Binarizes vector between two levels, min and max. The central value is (min_value+max_value)/2

    Parameters:
        vector: (numpy.array) array with values to binarize
        min_value (float): minimum value for binarization
        max_value (float): maximum value for binarization

    Returns:
        (numpy.array): binarized vector.
    """

    central_value = (min_value + max_value) / 2

    vector2 = deepcopy(vector)
    vector2[vector2 <= central_value] = min_value
    vector2[vector2 > central_value] = max_value
    return vector2


def discretize(u,
               kind='amplitude',
               num_levels=2,
               factor=1,
               phase0=0,
               new_field=True,
               matrix=False):
    """Discretize in a number of levels equal to num_levels.

    Parameters:
        kind (str): "amplitude" o "phase"
        num_levels (int): number of levels for the discretization
        factor (float): from the level, how area is binarized. if 1 everything is binarized,
        phase0 (float): *
        new_field (bool): if True returns new field
        matrix (bool): if True it returs a matrix

    Returns:
        scalar_fields_XY: if new_field is True returns scalar_fields_XY
    """

    if kind == 'amplitude':
        heights = np.linspace(0, 1, num_levels)
        posX = 256 / num_levels

        amplitude = get_amplitude(u)
        phase = get_phase(u)
        discretized_image = amplitude

        dist = factor * posX

        for i in range(num_levels):
            centro = posX / 2 + i * posX
            abajo = amplitude * 256 > centro - dist / 2
            arriba = amplitude * 256 <= centro + dist / 2
            Trues = abajo * arriba
            discretized_image[Trues] = centro / 256

        fieldDiscretizado = discretized_image * phase

    if kind == 'phase':
        ang = angle(get_phase(UnboundLocalError)) + phase0 + pi
        ang = ang % (2 * pi)
        amplitude = get_amplitude(u)

        heights = linspace(0, 2 * pi, num_levels + 1)

        dist = factor * (heights[1] - heights[0])

        discretized_image = exp(1j * (ang))

        for i in range(num_levels + 1):
            centro = heights[i]
            abajo = (ang) > (centro - dist / 2)
            arriba = (ang) <= (centro + dist / 2)
            Trues = abajo * arriba
            discretized_image[Trues] = exp(1j * centro)  # - pi

        Trues = (ang) > (centro + dist / 2)
        discretized_image[Trues] = exp(1j * heights[0])  # - pi

        # esto no haría falta, pero es para tener tantos levels
        # como decimos, no n+1 (-pi,pi)
        phase = angle(discretized_image) / pi
        phase[phase == 1] = -1
        phase = phase - phase.min()  # esto lo he puesto a última hora
        discretized_image = exp(1j * pi * phase)

        fieldDiscretizado = amplitude * discretized_image

        return fieldDiscretizado


def delta_kronecker(a, b):
    """Delta kronecker

    Parameters:
        a (np.float): number
        b (np.float): number

    Returns:
        1 if a==b and 0 if a<>b
    """

    if a == b:
        return 1
    else:
        return 0


def vector_product(A, B):
    """Returns the vector product between two vectors.

    Parameters:
        A (numpy.array): 3x1 vector array.
        B (numpy.array): 3x1 vector array.

    Returns:
        (numpy.array): 3x1 vector product array
    """

    Cx = A[1] * B[2] - A[2] * B[1]
    Cy = A[2] * B[0] - A[0] * B[2]
    Cz = A[0] * B[1] - A[1] * B[0]

    return np.array((Cx, Cy, Cz))


def dot_product(A, B):
    """Returns the dot product between two vectors.

    Parameters:
        A (numpy.array): 3x1 vector array.
        B (numpy.array): 3x1 vector array.

    Returns:
        (complex): 3x1 dot product
    """

    return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]


def divergence(E, r):
    """Returns the divergence of a field a given point (x0,y0,z0).

    Parameters:
        E (numpy.array): complex field
        r (numpy.array): 3x1 array with position r=(x,y,z).

    Returns:
        (float): Divergence of the field at (x0,y0,z0)
    """

    x0, y0, z0 = r

    dEx, dEy, dEz = np.gradient(E, x0[1] - x0[0], y0[1] - y0[0], z0[1] - z0[0])
    return dEx + dEy + dEz


def curl(E, r):
    """Returns the Curl of a field a given point (x0,y0,z0)

    Parameters:
        E (numpy.array): complex field
        r (numpy.array): 3x1 array with position r=(x,y,z).

    Returns:
        (numpy.array): Curl of the field at (x0,y0,z0)
    """

    x0, y0, z0 = r

    dEx, dEy, dEz = np.gradient(E, x0[1] - x0[0], y0[1] - y0[0], z0[1] - z0[0])
    componenteX = E[2] * dEy - E[1] * dEz
    componenteY = E[0] * dEz - E[2] * dEx
    componenteZ = E[1] * dEx - E[0] * dEy
    return [componenteX, componenteY, componenteZ]


def get_edges(x,
              f,
              kind_transition='amplitude',
              min_step=0,
              verbose=False,
              filename=''):
    """We have a binary mask and we obtain locations of edges.
    valid for litography engraving of gratings

    Parameters:
        x (float): position x
        f (numpy.array): Field. If real function, use 'amplitude' in kind_transition.
        kind_transition (str):'amplitude' 'phase' of the field where to get the transitions.
        min_step (float): minimum step for consider a transition
        verbose (bool): If True prints information about the process.
        filename (str): If not '', saves the data on files. filename is the file name.

    Returns:
        type_transition (numpy.array): array with +1, -1 with rasing or falling edges
        pos_transition (numpy.array): positions x of transitions
        raising (numpy.array): positions of raising
        falling (numpy.array): positions of falling
    """

    incr_x = x[1] - x[0]
    if kind_transition == 'amplitude':
        t = np.abs(f)
    elif kind_transition == 'phase':
        t = np.angle(f)
    diferencias = np.diff(t)
    t = np.concatenate((diferencias, np.array([0.])))

    raising = x[t > min_step] + .5 * incr_x
    falling = x[t < -min_step] + .5 * incr_x

    ones_raising = np.ones_like(raising)
    ones_falling = -np.ones_like(raising)

    pos_transitions = np.concatenate((raising, falling))
    type_transitions = np.concatenate((ones_raising, ones_falling))

    i_pos = np.argsort(pos_transitions)
    pos_transitions = pos_transitions[i_pos]
    type_transitions = type_transitions[i_pos]

    if verbose is True:
        print("position of transitions:")
        print("_______________________")
        print(np.array([pos_transitions, type_transitions]).T)
        print("\n\n")
        print("raising         falling:")
        print("_______________________")
        print(np.array([raising, falling]).T)

    if not filename == '':
        np.savetxt(
            "{}_pos_transitions.txt".format(filename),
            pos_transitions,
            fmt='%10.6f')
        np.savetxt(
            "{}_type_transitions.txt".format(filename),
            type_transitions,
            fmt='%10.6f')
        np.savetxt("{}_raising.txt".format(filename), raising, fmt='%10.6f')
        np.savetxt("{}_falling.txt".format(filename), falling, fmt='%10.6f')

    return pos_transitions, type_transitions, raising, falling


def cut_function(x, y, length, x_center=''):
    """ takes values of function inside (x_center+length/2: x_center+length/2)

        """
    if x_center in ('', None, []):
        x_center = (x[0] + x[-1]) / 2

    incr = length / 2
    left = x_center - incr
    right = x_center + incr

    i_min, _, _ = nearest(x, left)
    i_max, _, _ = nearest(x, right)

    y[0:i_min] = 0
    y[i_max::] = 0
    y[-1] = y[-2]

    return y


def fft_convolution2d(x, y):
    """ 2D convolution, using FFT

    Parameters:
        x (numpy.array): array 1 to convolve
        y (numpy.array): array 2 to convolve

    Returns:
        convolved function
    """
    return fftconvolve(x, y, mode='same')


def fft_convolution1d(x, y):
    """ 1D convolution, using FFT

    Parameters:
        x (numpy.array): array 1 to convolve
        y (numpy.array): array 2 to convolve

    Returns:
        convolved function
    """

    return fftconvolve(x, y, mode='same')


def fft_filter(x, y, normalize=False):
    """ 1D convolution, using FFT

    Parameters:
        x (numpy.array): array 1 to convolve
        y (numpy.array): array 2 to convolve

    Returns:
        convolved function
    """

    y = y / y.sum()

    return fftconvolve(
        x, y, mode='same') / fftconvolve(
            x, np.ones_like(y) / sum(y), mode='same')


def fft_correlation1d(x, y):
    """ 1D correlation, using FFT (fftconvolve)

    Parameters:
        x (numpy.array): array 1 to convolve
        y (numpy.array): array 2 to convolve

    Returns:
        numpy.array: correlation function
    """
    return fftconvolve(x, y[::-1], mode='same')


def fft_correlation2d(x, y):
    """Parameters:
        x (numpy.array): array 1 to convolve
        y (numpy.array): array 2 to convolve

    Returns:
        numpy.array: 2d correlation function
    """

    return fftconvolve(x, y[::-1, ::-1], mode='same')


def rotate_image(x, z, img, angle, pivot_point):
    """similar to rotate image, but not from the center but from the given

    Parameters:
        img (np.array): image to rotate
        angle (float): angle to rotate
        pivot_point (float, float): (z,x) position for rotation

    Returns:
        rotated image

    Reference:
        https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python point

    """

    # first get (i,j) pixel of rotation
    ipivotz, _, _ = nearest(z, pivot_point[0])
    ipivotx, _, _ = nearest(x, pivot_point[1])

    ipivot = [ipivotx, ipivotz]

    # rotates
    padX = [img.shape[1] - ipivot[0], ipivot[0]]
    padZ = [img.shape[0] - ipivot[1], ipivot[1]]
    imgP = np.pad(img, [padZ, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)

    return imgR[padZ[0]:-padZ[1], padX[0]:-padX[1]]


def cart2pol(x, y):
    """ cartesian to polar coordinate transformation.

    Parameters:
        x (np.array): x coordinate
        y (np.aray): y coordinate

    Returns:
        numpy.array: rho
        numpy.array: phi
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    """
    polar to cartesian coordinate transformation

    Parameters:
        rho (np.array): rho coordinate
        rho (np.aray): rho coordinate

    Returns:
        numpy.array: x
        numpy.array: y
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def fZernike(X, Y, n, m, radius=5 * mm):
    """Zernike function for aberration computation.

    Note:
        k>=l

        if k is even then l is even.
        if k is  odd then l is  odd.

        The first polinomial is the real part ant the second de imaginary part.


        * n     m        aberración
        * 0     0        piston
        * 1    -1        vertical tilt
        * 1     1        horizontal tilt
        * 2    -2        astigmatismo oblicuo
        * 2     0        desenfoque miopía si c>0 o desenfoque hipermetropía si c<0
        * 2     2        astigmatismo anormal si c>0 o astigmatismo normal si c<0
        * 3    -3        trebol oblicuo
        * 3    -1        coma vertical, c>0 empinamiento superior, c<0 emp. inferior
        * 3     1        como horizontal
        * 3     3        trebol horizontal
        * 4    -4        trebol de cuatro hojas oblicuo
        * 4    -2        astigmatismo secundario oblicuo
        * 4     0        esférica c>0 periferia más miópica que centro, c<0 periferia más hipertrópica que el centro
        * 4     2        astigmatismo secundario a favor o en contra de la regla
        * 4     4        trebol de cuatro hojas horizontal

    Reference:

        R. Navarro, J. Arines, R. Rivera "Direct and inverse discrete Zernike transform" Opt. Express 17(26) 24269

    """

    R = np.sqrt(X**2 + Y**2) / (radius)
    THETA = np.arctan2(X, Y)

    N = np.sqrt((n + 1) * (2 - delta_kronecker(m, 0)))

    Z = zeros(R.shape, dtype=np.float)
    for s in np.arange(0, (n - np.abs(m)) / 2 + 1):
        Z = Z + (-1)**s * R**(n - 2 * s) * factorial(
            np.abs(n - s)) / (factorial(np.abs(s)) * factorial(
                np.abs(round(0.5 * (n + np.abs(m)) - s))) * factorial(
                    np.abs(round(0.5 * (n - np.abs(m)) - s))))

    if m >= 0:
        fz1 = N * Z * np.cos(m * THETA)
    else:
        fz1 = N * Z * np.sin(np.abs(m) * THETA)

    fz1[R >= 1] = 0
    return fz1


def laguerre_polynomial_nk(x, n=4, k=5):
    """Auxiliar laguerre polinomial of orders n and k
        function y = LaguerreGen(varargin)
        LaguerreGen calculates the utilsized Laguerre polynomial L{n, alpha}
        This function computes the utilsized Laguerre polynomial L{n,alpha}.
        If no alpha is supplied, alpha is set to zero and this function
        calculates the "normal" Laguerre polynomial.


        Parameters:
        - n = nonnegative integer as degree level
        - alpha >= -1 real number (input is optional)

        The output is formated as a polynomial vector of degree (n+1)
        corresponding to MatLab norms (that is the highest coefficient
        is the first element).

        Example:
        - polyval(LaguerreGen(n, alpha), x) evaluates L{n, alpha}(x)
        - roots(LaguerreGen(n, alpha)) calculates roots of L{n, alpha}

        Calculation is done recursively using matrix operations for very fast
        execution time.

        Author: Matthias.Trampisch@rub.de
        Date: 16.08.2007
        Version 1.2

        References:
            Szeg: "Orthogonal Polynomials" 1958, formula (5.1.10)

        """

    f = factorial
    summation = np.zeros_like(x, dtype=float)
    for m in range(n + 1):
        summation = summation + (-1)**m * f(n + k) / (
            f(n - m) * f(k + m) * f(m)) * x**m
    return summation


def get_k(x, flavour='-'):
    """provides k vector from x vector. Two flavours are provided (ordered + or disordered - )

    Parameters:
        x (np.array): x array
        flavour (str): '+' or '-'

    Returns:
        kx (np.array): k vector

    Todo:
        Check
    """

    num_x = x.size
    if flavour == '-':
        size_x = x[-1] - x[0]

        kx1 = np.linspace(0, num_x / 2 + 1, int(num_x / 2))
        kx2 = np.linspace(-num_x / 2, -1, int(num_x / 2))
        kx = (2 * np.pi / size_x) * np.concatenate((kx1, kx2))

    elif flavour == '+':
        dx = x[1] - x[0]
        kx = 2 * np.pi / (num_x * dx) * (range(-int(num_x / 2), int(
            num_x / 2)))

    return kx


def filter_edge_1D(x, size=1.1, exponent=32):
    """function 1 at center and reduced at borders. For propagation algorithms

    Parameters:
        x (np.array): position
        size (float): related to relative position of x
        exponent (integer): related to shape of edges
    Returns:
        np.array: function for filtering
    """

    # num_x = len(x)
    x_center = (x[-1] + x[0]) / 2
    Dx = size * (x[-1] - x[0])
    return np.exp(-(2 * (x - x_center) / (Dx))**np.abs(exponent))


def filter_edge_2D(x, y, size=1.1, exponent=32):
    """function 1 at center and reduced at borders. For propagation algorithms

    Parameters:
        x (np.array): x position
        y (np.array): y position
        size (float): related to relative position of x and y
        exponent (integer): related to shape of edges
    Returns:
        np.array: function for filtering
    """

    # num_x = len(x)
    x_center = (x[-1] + x[0]) / 2
    y_center = (y[-1] + y[0]) / 2
    Dx = size * (x[-1] - x[0])
    Dy = size * (y[-1] - y[0])

    X, Y = np.meshgrid(x, y)

    exp1 = np.exp(-(2 * (X - x_center) / (Dx))**np.abs(exponent))
    exp2 = np.exp(-(2 * (Y - y_center) / (Dy))**np.abs(exponent))

    return exp1 * exp2

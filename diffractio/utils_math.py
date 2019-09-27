#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Common functions to classes """

from copy import deepcopy

import numpy as np
import scipy.ndimage as ndimage
from numpy import angle, array, exp, linspace, ones_like, pi, sqrt, zeros
from scipy.signal import fftconvolve


# tested
def distance(x1, x2):
    """
    Compute distance between two vectors.

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


def ndgrid(*args, **kwargs):
    """n-dimensional gridding like Matlab's NDGRID

    The input *args are an arbitrary number of numerical sequences,
    e.g. lists, arrays, or tuples.
    The i-th dimension of the i-th output argument
    has copies of the i-th input argument.

    Optional keyword argument:
    same_dtype : If False (default), the result is an ndarray.
                 If True, the result is a lists of ndarrays, possibly with
                   different dtype. This can save space if some *args
                   have a smaller dtype than others.

    Typical usage:
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
    """Passes the amplitude of a complex field to phase. Previous phase is removed.
    $u = A e^{i \phi}  -> e^(i abs(amp))$

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


def normalize_field(u, kind='intensity'):
    """Normalize the field

    Parameters:
        kind (str): 'intensity' 'area'

    Todo:
        pass to utils
    """

    if kind == 'intensity':
        intensity = np.abs(u**2)
        maximum = sqrt(intensity.max())
        u = u / maximum
    if kind == 'area':
        intensity = np.abs(u**2)
        maximum = intensity.sum()
        u = u / maximum

        return u


def binarize(vector, min_value=0, max_value=1):
    """Binarizes vector between two levels, min and max.
    The central value is (min_value+max_value)/2

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
        factor (float): from the level, how area is binarized
            if 1 everything is binarized,
        phase0 (float): *
        new_field (bool): if True returns new field
        matrix (bool): if True it returs a matrix

    Returns:
        scalar_fields_XY: if new_field is True returns scalar_fields_XY

    Todo:
        Check and pass to utils
    """

    if kind == 'amplitude':
        heights = linspace(0, 1, num_levels)
        posX = 256 / num_levels

        amplitude = get_amplitude(matrix=True, new_field=False)
        phase = get_phase(matrix=True, new_field=False)
        imageDiscretizada = amplitude

        dist = factor * posX
        print("dist:", dist)

        for i in range(num_levels):
            centro = posX / 2 + i * posX
            abajo = amplitude * 256 > centro - dist / 2
            arriba = amplitude * 256 <= centro + dist / 2
            Trues = abajo * arriba
            imageDiscretizada[Trues] = centro / 256
            # heights[i]+posX/(256*2)
            # falta compute el porcentaje de height

        fieldDiscretizado = imageDiscretizada * phase

    if kind == 'phase':
        ang = angle(get_phase(matrix=True, new_field=False)) + phase0 + pi
        ang = ang % (2 * pi)
        amplitude = get_amplitude(matrix=True, new_field=False)

        heights = linspace(0, 2 * pi, num_levels + 1)
        # no hay tantos levels, pero es para buscar los centros
        # heights=heights[0:-1]
        # anchuras = 2 * pi / num_levels
        # cortes = heights + anchuras / 2
        dist = factor * (heights[1] - heights[0])

        # print "heights: ", heights
        # print "cortes: ", cortes

        imageDiscretizada = exp(1j * (ang))

        for i in range(num_levels + 1):
            centro = heights[i]
            abajo = (ang) > (centro - dist / 2)
            arriba = (ang) <= (centro + dist / 2)
            Trues = abajo * arriba
            imageDiscretizada[Trues] = exp(1j * (centro))  # - pi

        Trues = (ang) > (centro + dist / 2)
        imageDiscretizada[Trues] = exp(1j * (heights[0]))  # - pi

        # esto no haría falta, pero es para tener tantos levels
        # como decimos, no n+1 (-pi,pi)
        phase = angle(imageDiscretizada) / pi
        phase[phase == 1] = -1
        phase = phase - phase.min()  # esto lo he puesto a última hora
        imageDiscretizada = exp(1j * pi * phase)

        fieldDiscretizado = amplitude * imageDiscretizada

        return fieldDiscretizado


def delta_kronecker(a, b):
    """Delta kronecker"""
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
    """Returns the divergence of a field a given point (x0,y0,z0)
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


def cut_function(x, y, longitud):
    """
        tenemos una function y(x) que tiene ciertos values
        solamente dejamos fuera de 0 aquellos que esten dentro de longitud
        """
    x_central = (x[0] + x[-1]) / 2
    incr = longitud / 2
    left = x_central - incr
    right = x_central + incr

    i_min, _, _ = nearest(x, left)
    i_max, _, _ = nearest(x, right)

    y[0:i_min] = 0
    y[i_max::] = 0
    y[-1] = y[-2]

    return y


def muestrear(xsampling, data):
    """devuelve un array de points, ysampling,
     donde vale 0 si no es nearest y 1 si lo es.
        Esta function es valida para luego hacer las convoluciones

        Parameters:
        * xsampling : array de points que son las posiciones del sampling
        * xdata    : array  de points de las posiciones de los 'sensors'

        outputs:
        * ysampling : array con el mismo length de ysampling con values 0 o 1

        mejoras:
        * he metido un bucle for que creo que habria que remove
        * incluir n dimensiones
        """

    ysampling = np.zeros(xsampling.shape, dtype=float)

    imenores, values, distances = nearest2(xsampling, data)  # @UnusedVariable
    for i in imenores:
        ysampling[int(i)] = 1.
    return ysampling


# def sampling(u, x0=0, x1=1, num_data=1000):
#     """
#     Esta function pone 1 donde hay un dato y 0 en el resto
#     tambien manda una function identica para los values de x
#     """
#
#     xMuestreo = np.linspace(x0, x1, num_data)  # array con el value de x
#     xComb = np.zeros(len(xMuestreo))      # array con 1 en posiciones
#     yComb = np.zeros(len(xMuestreo))      # array con y(i) en posiciones
#
# for idato in range(len(self.x)):
# #             imenor, value, distance = nearest(xMuestreo, self.x[idato])
# #             xComb[imenor] = 1
# #             yComb[imenor] = self.y[idato]
#
#     imenores, values, distances = nearest2(xMuestreo, self.x)
#     xComb[imenores] = np.ones(imenores.shape)
#     yComb[imenores] = self.y
#
#     return xMuestreo, xComb, yComb


def fft_convolution2d(x, y):
    """ 2D convolution, using FFT"""
    return fftconvolve(x, y, mode='same')


def fft_convolution1d(x, y):
    """ 1D convolution, using FFT """
    return fftconvolve(x, y, mode='same')


def fft_correlation1d(x, y):
    return fftconvolve(x, y[::-1], mode='same')


def fft_correlation2d(x, y):
    return fftconvolve(x, y[::-1, ::-1], mode='same')


def rotate_image(x, z, img, angle, pivot_point):
    """similar to rotate image, but not from the center but from the given point
    https://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python

    Parameters:
        img (np.array): image to rotate
        angle (float): angle to rotate
        pivot_point (float, float): (z,x) position for rotation

    Returns:
        rotated image
    """

    # first get (i,j) pixel of rotation

    ipivotz, _, _ = nearest(z, pivot_point[0])
    ipivotx, _, _ = nearest(x, pivot_point[1])

    ipivot = [ipivotz, ipivotx]

    # rotates

    padX = [img.shape[1] - ipivot[0], ipivot[0]]
    padZ = [img.shape[0] - ipivot[1], ipivot[1]]
    imgP = np.pad(img, [padZ, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)

    return imgR[padZ[0]:-padZ[1], padX[0]:-padX[1]]


def cart2pol(x, y):
    """
    cartesian to polar coordinate transformation
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    polar to cartesian coordinate transformation
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


#
# def fft_convolution2d_shen_proposal(u1, u2, new_field=True, verbose=True):
#     """Convolution procedure Applied Optics vol 45 num 6 pp. 1102-1110 (2006)
#
#     With field of size N*M, the result of propagation is also a field N*M.
#
#     Parameters:
#         u1, u2 (numpy.array): fields
#         new_field (bool): if False the computation goes to self.u
#                           if True a new instance is produced
#         verbose (bool): if True it writes to shell
#
#     Returns:
#         if New_field is True: Scalar_field_X
#         else None
#     """
#
#     nx, ny = u1.shape
#
#     # matrix W para integración simpson
#     # he tenido problemas porque en shen viene para matrices cuadradas
#     # y yo admito matrices rectangulares. pero he solucionado.
#     a = [2, 4]
#     num_repx = int(round((nx) / 2) - 1)
#     num_repy = int(round((ny) / 2) - 1)
#     # print( num_repx, num_repy)
#     bx = array(a * num_repx)
#     by = array(a * num_repy)
#     cx = concatenate(((1, ), bx, (2, 1))) / 3.
#     cy = concatenate(((1, ), by, (2, 1))) / 3.
#
#     if float(nx) / 2 == round(nx / 2):  # es par
#         i_centralx = num_repx + 1
#         cx = concatenate((cx[:i_centralx], cx[i_centralx + 1:]))
#     if float(ny) / 2 == round(ny / 2):  # es par
#         i_centraly = num_repy + 1
#         cy = concatenate((cy[:i_centraly], cy[i_centraly + 1:]))
#
#     d1x = matrix(cx)
#     d1y = matrix(cy)
#     W = array(d1y.T * d1x)
#     # W=1
#
#     U1 = zeros((2 * ny - 1, 2 * nx - 1), dtype=complex)
#     U1[0:ny, 0:nx] = array(W * u1)
#
#     # calculo de la transformada de Fourier
#     S = ifft2(fft2(U1) * fft2(u2))
#     # transpose cambiado porque daba problemas para matrices no cuadradas
#     return S[ny - 1:, nx - 1:]  # hasta el final

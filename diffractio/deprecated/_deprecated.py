# !/usr/bin/env python3

""" Deprecated functions """

# utils_math

import numpy as np


def ndgrid_deprecated(*args, **kwargs):
    """n-dimensional gridding like Matlab's NDGRID.

    Args:
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
            np.array([[[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]],
                    [[1, 1, 1, 1],
                            [1, 1, 1, 1],
                            [1, 1, 1, 1]]])
        >>> Y
            np.array([[[2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [4, 4, 4, 4]],
                    [[2, 2, 2, 2],
                            [3, 3, 3, 3],
                            [4, 4, 4, 4]]])
        >>> Z
            np.array([[[5, 6, 7, 8],
                            [5, 6, 7, 8],
                            [5, 6, 7, 8]],
                    [[5, 6, 7, 8],
                            [5, 6, 7, 8],
                            [5, 6, 7, 8]]])

        With an unpacked argument list:

        >>> V = [[0, 1], [2, 3, 4]]

        >>> ndgrid_deprecated(*V) # an np.array of two arrays with shape (2, 3)
            np.array([[[0, 0, 0],
                [1, 1, 1]],
                [[2, 3, 4],
                [2, 3, 4]]])

        For input vectors of different data kinds,
        same_dtype=False makes ndgrid_deprecated()
        return a list of arrays with the respective dtype.
        >>> ndgrid_deprecated([0, 1], [1.0, 1.1, 1.2], same_dtype=False)
        [np.array([[0, 0, 0], [1, 1, 1]]),
        np.array([[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]])]

        Default is to return a single np.array.

        >>> ndgrid_deprecated([0, 1], [1.0, 1.1, 1.2])
            np.array([[[ 0. ,  0. ,  0. ], [ 1. ,  1. ,  1. ]],
                [[ 1. ,  1.1,  1.2], [ 1. ,  1.1,  1.2]]])
    """
    same_dtype = kwargs.get("same_dtype", True)
    V = [np.array(v) for v in args]  # ensure all input vectors are arrays
    shape = [len(v) for v in args]  # common shape of the outputs
    result = []
    for i, v in enumerate(V):
        # reshape v so it can broadcast to the common shape
        # http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        zero = np.zeros(shape, dtype=v.dtype)
        thisshape = np.ones_like(shape)
        thisshape[i] = shape[i]
        result.append(zero + v.reshape(thisshape))
    if same_dtype:
        return np.array(result)  # converts to a common dtype
    else:
        return result  # keeps separate dtype for each output


# Utils_math
def get_k_deprecated(x, flavour='-'):
    """provides k vector from x vector. Two flavours are provided (ordered + or disordered - )

    Args:
        x (np.array): x np.array
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


# utils_optics
# def normalize(u, kind='intensity'):
#     """Normalizes a field to have intensity or amplitude, etc. 1

#     Args:
#         u (numpy.array): optical field (comes usually form field.u)
#         kind (str): 'intensity, 'amplitude', 'logarithm'... other.. Normalization technique

#     Returns
#         u (numpy.array): normalized optical field
#     """

#     if kind == 'intensity':
#         intensity_max = (np.abs(u)).max()
#         u = u / intensity_max
#     elif kind == 'amplitude':
#         amplitude_max = np.sqrt(np.abs(u)).max()
#         u = u / amplitude_max

#     return u

# Util Optics
# def normalize_vector_deprecated(u):
#     """Normalizes a vector to have intensity or amplitude, etc. 1

#     Args:
#         u (numpy.array): vector (last dimension should have size 2 or 3)

#     Returns
#         u (numpy.array): normalized optical field
#     """
#     return u / np.linalg.norm(u)


# vector_fields_Xy

    # def mask_circle(self, r0=(0., 0.), radius=0.):
    #     """Mask vector field using a circular mask.

    #     Args:
    #         r0 (float, float): center of mask.
    #         radius (float, float): radius of mask
    #     """

    #     if isinstance(radius, (float, int, complex)):
    #         radiusx, radiusy = (radius, radius)
    #     else:
    #         radiusx, radiusy = radius
    #     radius = (radiusx, radiusy)

    #     if radiusx * radiusy > 0:
    #         radius_x = (self.x[-1] - self.x[0]) / 2
    #         radius_y = (self.y[-1] - self.y[0]) / 2
    #         radius = (radius_x, radius_y)

    #     elif radius in (None, '', []):
    #         return

    #     elif isinstance(radius, (float, int, complex)):
    #         radius = (radius, radius)

    #     if r0 in (0, None, '', []):
    #         r0_x = (self.x[-1] + self.x[0]) / 2
    #         r0_y = (self.y[-1] + self.y[0]) / 2
    #         r0 = (r0_x, r0_y)

    #     if radiusx * radiusy > 0:
    #         t1 = Scalar_mask_XY(x=self.x, y=self.y, wavelength=self.wavelength)
    #         t1.circle(r0=r0, radius=radius, angle=0 * degrees)
    #         self.Ex = t1.u * self.Ex
    #         self.Ey = t1.u * self.Ey
    #         self.Ez = t1.u * self.Ez

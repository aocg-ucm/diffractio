# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" General purpose optics functions """

import pandas as pd

from . import degrees, np, plt
from .utils_math import (
    fft_convolution1d,
    fft_convolution2d,
    find_extrema,
    ndgrid_deprecated,
    nearest,
)


def roughness_1D(x, t, s, kind="normal"):
    """Rough surface, 1D

    Parameters:
        x (numpy.array): array with x positions
        t (float): correlation lens
        s (float): std of roughness
        kind (str): 'normal', 'uniform'

    Returns:
        (numpy.array) Topography of roughnness in microns.

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.

    """

    width = x[-1] - x[0]
    dx = x[1] - x[0]

    # Surface parameters

    L_ancho = width / (2 * dx)
    M = round(4 * t / (np.sqrt(2) * dx))

    N_ancho = int(np.floor(L_ancho + M))

    desp_ancho = np.arange(-M, M + 1)

    desp_ancho = desp_ancho * dx
    pesos = np.exp(-2 * (desp_ancho**2 / t**2))

    pesos = np.abs(pesos / np.sqrt((pesos**2).sum()))

    if kind == "normal":
        h_no_corr = s * np.random.randn(2 * N_ancho + 1)
        h_corr = fft_convolution1d(h_no_corr, pesos)
        h_corr = h_corr[0 : len(x)]
    elif kind == "uniform":
        h_corr = s * (np.random.rand(len(x)) - 0.5)
    return h_corr


def roughness_2D(x, y, t, s):
    """Rough surface, 2D

    Parameters:
        x (numpy.array): x positions
        y (numpy.array): y positions
        t (float, float): (tx, ty), correlation length of roughness
        s (float): std of heights

    Returns:
        (numpy.array) Topography of roughnness in microns.

    Example:
        roughness(t=(50 * um, 25 * um), s=1 * um)

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.
    """

    if isinstance(t, (float, int, complex)):
        t = (t, t)

    tx, ty = t

    width = x[-1] - x[0]
    largo = y[-1] - y[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    L_ancho = width / (2 * dx)
    L_largo = largo / (2 * dy)
    Mx = round(4 * tx / (np.sqrt(2) * dx))
    My = round(4 * ty / (np.sqrt(2) * dy))

    N_ancho = int(np.floor(L_ancho + Mx))
    N_largo = int(np.floor(L_largo + My))

    desp_ancho, desp_largo = np.meshgrid(np.arange(-Mx, Mx + 1), np.arange(-My, My + 1))
    desp_ancho = desp_ancho * dx
    desp_largo = desp_largo * dy

    pesos = np.exp(-2 * (desp_ancho**2 / tx**2 + desp_largo**2 / ty**2))
    pesos = np.abs(pesos / np.sqrt((pesos**2).sum()))

    h_no_corr = s * np.random.randn(2 * N_ancho + 1, 2 * N_largo + 1)
    h_corr = fft_convolution2d(h_no_corr, pesos)
    h_corr = h_corr[0 : len(x), 0 : len(y)]
    return h_corr


def beam_width_1D(u, x, remove_background=None):
    """One dimensional beam width, according to D4σ or second moment width.

    Parameters:
        u (np.array): field (not intensity).
        x (np.array): x

    Returns:
        (float): width
        (float): x_mean

    References:
        https://en.wikipedia.org/wiki/Beam_diameter
    """

    intensity = np.abs(u) ** 4

    if remove_background is True:
        intensity = intensity - intensity - min()

    P = (intensity).sum()
    x_mean = (intensity * x).sum() / P
    x2_mean = (intensity * (x - x_mean) ** 2).sum() / P
    width_x = 4 * np.sqrt(x2_mean)
    return width_x, x_mean


def width_percentage(x, y, percentage=0.5, verbose=False):
    """beam width (2*sigma) given at a certain height from maximum

    Parameters:
        x (np.array): x
        y (np.array): y
        percentage (float): percentage of height. For example: 0.5

    Returns:
        (float): width, width of at given %
        (list): x_list: (x[i_left], x[i_max], x[i_right])
        (list): x_list: (i_left, i_max, i_right)

    Notes:
        y=np.exp(-x**2/(s**2))  percentage=1/e -> width = 2*s
        y=np.exp(-x**2/(s**2))  percentage=1/e**4 -> width = 4*s
        y=np.exp(-x**2/(2*s**2))  percentage=1/e**2 =  -> width = 4*s

    """

    maximum = y.max()
    level = percentage * maximum
    i_max = np.argmax(y)

    if i_max == 0:
        i_left = 0
        print("beam width out of range")
    else:
        i_left, _, _ = nearest(y[0:i_max], level)

    if i_max == len(y) - 1:
        i_right = len(y) - 1
        print("beam width out of range")
    else:
        i_right, _, _ = nearest(y[i_max:-1], level)
        i_right = i_right + i_max

    if verbose is True:
        print(i_left, i_max, i_right)

    width = x[i_right] - x[i_left]

    x_list = (x[i_left], x[i_max], x[i_right])
    i_list = (i_left, i_max, i_right)

    return width, x_list, i_list


def beam_width_2D(x, y, intensity, remove_background=False, has_draw=False):
    """2D beam width, ISO11146 width


    Parameters:
        x (np.array): 1d x
        y (np.array): 1d y
        intensity (np.array):  intensity

    Returns:
        (float): dx width x
        (float): dy width y
        (float): principal_axis, angle
        (str): (x_mean, y_mean, x2_mean, y2_mean, xy_mean), Moments

    References:

        * https://en.wikipedia.org/wiki/Beam_diameter

        * http://www.auniontech.com/ueditor/file/20170921/1505982360689799.pdf


    """
    X, Y = np.meshgrid(x, y)
    if remove_background is True:
        intensity = intensity - intensity - min()

    P = intensity.sum()
    x_mean = (intensity * X).sum() / P
    y_mean = (intensity * Y).sum() / P
    x2_mean = (intensity * (X - x_mean) ** 2).sum() / P
    y2_mean = (intensity * (Y - y_mean) ** 2).sum() / P
    xy_mean = (intensity * (X - x_mean) * (Y - y_mean)).sum() / P
    # gamma = (x2_mean - y2_mean) / np.abs(x2_mean - y2_mean + 1e-16)
    gamma = np.sign(x2_mean - y2_mean + 0.0000000001)
    rt = np.sqrt((x2_mean - y2_mean) ** 2 + 4 * xy_mean**2)
    dx = 2 * np.sqrt(2) * np.sqrt(x2_mean + y2_mean + gamma * rt)
    dy = 2 * np.sqrt(2) * np.sqrt(x2_mean + y2_mean - gamma * rt)

    # print(gamma)
    # print(rt)
    # print(x2_mean, y2_mean, rt, dx, dy)

    principal_axis = 0.5 * np.arctan2(2 * xy_mean, x2_mean - y2_mean)

    if has_draw is True:
        from matplotlib.patches import Ellipse

        from .scalar_fields_XY import Scalar_field_XY

        u0 = Scalar_field_XY(x, y, 1)
        u0.u = np.sqrt(intensity)
        u0.draw()
        ellipse = Ellipse(
            xy=(x_mean, y_mean), width=dy, height=dx, angle=-principal_axis / degrees
        )

        ax = plt.gca()
        ax.add_artist(ellipse)
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.75)
        ellipse.set_facecolor("none")
        ellipse.set_edgecolor([1, 1, 1])
        ellipse.set_linewidth(3)

    return dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean, xy_mean)


def refractive_index(filename, wavelength, raw=False, has_draw=True):
    """gets refraction index from https://refractiveindex.info .

    * Files has to be converted to xlsx format.
    * n and k checks has to be activated.

    Parameters:
        filename (str): xlsx file
        wavelength (float): wavelength in microns, example, 0.6328.
        raw (bool): if True returns all the data in file.
        has_draw (bool): draw the data from the file

    Returns:
        if raw is False (float, float): n, k  from each wavelength
        if raw is True  (np.array, np.array): n,k for wavelengths in file


    """
    data = pd.read_excel(filename)

    wavelengths = data["Wavelength, µm"].values.astype(float)
    n = data["n"].values.astype(float)
    kappa = data["k"].values.astype(float)

    if has_draw is True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("wavelengths (nm)")
        ax1.plot(wavelengths, n, "r", label="n")
        ax1.set_ylabel("n", color="r")
        ax1.tick_params(axis="y", labelcolor="r")
        ax2 = ax1.twinx()
        ax2.plot(wavelengths, kappa, "b", label=r"$\kappa$")
        ax2.set_ylabel(r"$\kappa$", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
        fig.tight_layout()
        fig.legend()

    if raw is True:
        return wavelengths, n, kappa

    else:
        z_n = np.polyfit(wavelengths, n, 6)
        z_kappa = np.polyfit(wavelengths, kappa, 6)

        f_n = np.poly1d(z_n)
        f_kappa = np.poly1d(z_kappa)

        return f_n(wavelength), f_kappa(wavelength)


def FWHM1D(x, intensity, percentage=0.5, remove_background=None, has_draw=False):
    """FWHM1D

    remove_background = 'min', 'mean', None"""

    if remove_background == "mean":
        I_background = intensity.mean()
    elif remove_background == "min":
        I_background = intensity.min()
    else:
        I_background = 0

    intensity = intensity - I_background

    if type(remove_background) is float:
        intensity[intensity < remove_background * intensity.max()] = 0

    delta_x = x[1] - x[0]
    amp_max = intensity.max()
    amp_med = amp_max * percentage
    i_max = np.where(intensity == amp_max)
    i_max = int(i_max[0][0])
    left = intensity[0:i_max]
    right = intensity[i_max::]

    i_left, _, distance_left = nearest(left, percentage * amp_max)
    slope_left = (intensity[i_left + 1] - intensity[i_left]) / delta_x

    i_right, _, distance_right = nearest(right, percentage * amp_max)
    slope_right = (
        intensity[i_max + i_right] - intensity[i_max + i_right - 1]
    ) / delta_x

    i_right = i_right + i_max

    x_right = i_right * delta_x - distance_right / slope_right
    x_left = i_left * delta_x - distance_left / slope_left

    FWHM_x = x_right - x_left

    amp_max = amp_max + I_background
    amp_med = amp_med + I_background

    intensity = intensity + I_background

    if has_draw is True:
        if remove_background is True:
            intensity = intensity + intensity.min()
        plt.figure()

        plt.plot(x, intensity, "k", lw=2)
        plt.plot([x[0], x[-1]], [amp_max, amp_max], "r--")
        plt.plot([x[0], x[-1]], [amp_med, amp_med], "r--")

        plt.plot(x[i_max], intensity[i_max], "ro", ms=8)
        plt.plot(x[int(i_right)], intensity[int(i_left)], "ro", ms=8)
        plt.plot(x[int(i_left)], intensity[int(i_right)], "ro", ms=8)
        plt.ylim(ymin=0)
        plt.xlim(x[0], x[-1])

    return np.squeeze(FWHM_x)


def FWHM2D(
    x, y, intensity, percentage=0.5, remove_background="None", has_draw=False, xlim=None
):
    """TODO: perform profiles at several angles and fit to a ellipse.
    Get dx, dy, angle, x_center, y_center"""
    # Ix = intensity.mean(axis=0)
    # Iy = intensity.mean(axis=1)

    i_pos, _, I_max = find_extrema(intensity.transpose(), x, y, kind="max")

    Ix = intensity[:, i_pos[0, 1]]
    Iy = intensity[i_pos[0, 0], :]

    # print(x.shape, Iy.shape)
    FWHM_x = FWHM1D(x, Ix, percentage, remove_background, has_draw=has_draw)
    if has_draw is True:
        if xlim is not None:
            plt.xlim(xlim)

    # print(y.shape, Iy.shape)
    FWHM_y = FWHM1D(y, Iy, percentage, remove_background, has_draw=has_draw)
    if has_draw is True:
        if xlim is not None:
            plt.xlim(xlim)

    return FWHM_x, FWHM_y


def DOF(z, widths, w_factor=np.sqrt(2), w_fixed=0, has_draw=False, verbose=False):
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

    if w_fixed == 0:
        beam_waist = widths.min()
        i_w0 = np.where(widths == beam_waist)
        i_w0 = int(i_w0[0][0])

    else:
        beam_waist = w_fixed
        i_w0, _, _ = nearest(widths, beam_waist)

    left = widths[0:i_w0]
    right = widths[i_w0::]

    i_left, _, distance_left = nearest(left, w_factor * beam_waist)

    i_right, _, distance_right = nearest(right, w_factor * beam_waist)

    z_rayleigh = z[i_right + i_w0] - z[i_left]

    if verbose:
        print(i_w0, widths[i_w0])
        print(z_rayleigh)

        print(widths[i_right + i_w0], z[i_right + i_w0])
        print(widths[i_left], z[i_left])

    if has_draw:
        plt.figure()

        plt.plot(z, widths, "k", lw=2)
        plt.plot(z, -widths, "k", lw=2)
        plt.plot(z, np.zeros_like(z), "k-.", lw=2)

        plt.plot([z[i_left], z[i_left]], [-widths[i_left], widths[i_left]], "r--")
        plt.plot(
            [z[i_right + i_w0], z[i_right + i_w0]],
            [-widths[i_right + i_w0], widths[i_right + i_w0]],
            "r--",
        )
        plt.annotate(
            text="",
            xy=(z[i_left], -widths[i_right + i_w0]),
            xytext=(z[i_right + i_w0], -widths[i_right + i_w0]),
            arrowprops=dict(arrowstyle="<->"),
        )
        plt.text(z[i_w0], -widths.mean(), "$z_{R}$", fontsize=18)
        plt.xlim(z[0], z[-1])
        plt.ylim(-widths.max(), widths.max())

    return z_rayleigh, beam_waist, np.array([z[i_left], z[i_w0], z[i_right + i_w0]])


def detect_intensity_range(
    x, intensity, percentage=0.95, has_draw=True, logarithm=True
):
    """Determines positions x_min, x_max where intensity of the beam is percentage

    Parameters:
        x (np.array): x positions
        intensity (np.array): Intensity of the 1D beam
        percentage (float): value 0-1 representing the percentage of intensity between area
        has_draw (bool): if True draws the field an the range
        logarithm (bool): when has_draw, draws logarithm or normal intensity

    Returns:
        (float, float): positions (x_min, right) where intensity beam is enclosed at %.

    """

    I_cum = intensity.cumsum()

    pc = percentage + (1 - percentage) / 2
    Icum_min = (1 - pc) * I_cum.max()
    Icum_max = I_cum.max() * pc

    I_min = intensity.min()
    I_max = intensity.max()

    i_min, _, _ = nearest(I_cum, Icum_min)
    i_max, _, _ = nearest(I_cum, Icum_max)

    x_min = x[i_min]
    x_max = x[i_max]

    if has_draw is True:
        _, ax = plt.subplots()

        if logarithm is True:
            I2 = np.log(intensity + 1)
            I_min2 = np.log(I_min + 1)
            I_max2 = np.log(I_max + 1)

            I2 = I2 / I2.max()
            I_max2 = I_max2 / I_max2.max()
        else:
            I2 = intensity
            I_min2 = I_min
            I_max2 = I_max

        ax.plot(x, I2, c="r", alpha=1, lw=4)

        x_bordes = [x_min, x_max, x_max, x_min, x_min]
        y_bordes = [I_min2, I_min2, I_max2, I_max2, I_min2]

        ax.fill(x_bordes, y_bordes, c="r", alpha=0.25)

    return x_min, x_max


def MTF_ideal(
    frequencies, wavelength, diameter, focal, kind, verbose=False, has_draw=False
):
    """Determines the ideal MTF of a lens.

    References:
        https://www.edmundoptics.com/resources/application-notes/optics/introduction-to-modulation-transfer-function/

        https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf

    Parameters:
        frequencies (numpy.array): array with frequencies in *lines/mm*
        wavelength (float): wavelength of incoming light beam
        diameter (float): diameter of lens
        focal (float): focal distance of lens
        kind (float): '1D', '2D'
        verbose (bool): if True displays limit frequency of the lens

    Returns:
        (numpy.array) MTF: Normalized MTF of ideal lens
        (float) frequency_max: maximum frequency of the lens
    """

    F_number = focal / diameter
    frequency_max = 1000.0 / (wavelength * F_number)  # porque mido en micras
    fx_norm = np.abs(frequencies / frequency_max)

    if kind == "1D":
        MTF = 1 - np.abs(fx_norm)
        MTF[fx_norm > 1] = 0

    elif kind == "2D":
        fx2 = np.arccos(fx_norm)
        MTF = np.real(2 / np.pi * (fx2 - np.cos(fx2) * np.sin(fx2)))

        # otra definición: https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf
        # MTF = np.real(2/np.pi*(np.arccos(fx_norm)-fx_norm*np.sqrt(1-fx_norm**2)))

        # isH1 = MTF > 1
        # MTF[isH1] = 2 - MTF[isH1]

    if verbose is True:
        print("frecuencia de bin_level = {:4.2f} lineas/mm".format(frequency_max))

    if has_draw is True:
        plt.figure()
        plt.plot(frequencies, MTF, "k")
        plt.xlabel("$f_x (mm^{-1})$", fontsize=18)
        plt.ylabel("MTF", fontsize=18)

    return MTF, frequency_max


def lines_mm_2_cycles_degree(lines_mm, focal):
    """Pasa líneas por mm a cyclos/grado, más tipico de ojo
    Infor saca estos cálculos 181022
    Parameters:
        lines_mm (numpy.array or float): lines_per_mm
        focal (float): focal of lens
    """

    frec_cycles_deg = 180 * focal * lines_mm / np.pi

    return frec_cycles_deg


def MTF_parameters(MTF, MTF_ideal, lines_mm=50, verbose=False):
    """MTF parameters: strehl_ratio, mtf_50_ratio, freq_50_real, freq_50_ideal

    References:
        https://www.edmundoptics.com/resources/application-notes/optics/introduction-to-modulation-transfer-function/strehl_ratio

    frequencies of mtf ar given since both MTF can have different steps
    MTF:


    Parameters:
        MTF (N,2 numpy.array): (freq, MTF) of system in lines/mm
        MTF_ideal (M,2 numpy.array): (freq, MTF) of ideal system in lines/mm
        lines_mm (float): (0-1) Height of MTF for ratios

    Returns:
        (float): strehl_ratio
        (float): MTF_ratio at freq_obs height
        (float): frequency at freq_obs of MTF
        (float): frequency at freq_obs of MTF_ideal
    """

    fx_real, mtf_real = MTF
    fx_ideal, mtf_ideal = MTF_ideal

    i_0_real, _, _ = nearest(fx_real, 0)
    i_0_ideal, _, _ = nearest(fx_ideal, 0)

    dx_real = fx_real[1] - fx_real[0]
    dx_ideal = fx_ideal[1] - fx_ideal[0]

    mtf_real = mtf_real[i_0_real::]
    mtf_ideal = mtf_ideal[i_0_ideal::]

    fx_real = fx_real[i_0_real::]
    fx_ideal = fx_ideal[i_0_ideal::]

    # STREHL RATIO
    strehl_ratio = (mtf_real.sum() * dx_real) / (mtf_ideal.sum() * dx_ideal)

    # MTF at 50 (u other) lines/mm
    imenor_ideal, _, _ = nearest(fx_ideal, lines_mm)
    imenor_real, _, _ = nearest(fx_real, lines_mm)

    freq_50_ideal = np.abs(mtf_ideal[imenor_ideal])
    freq_50_real = np.abs(mtf_real[imenor_real])

    mtf_50_ratio = freq_50_real / freq_50_ideal

    if verbose is True:
        print(" MTF Parameters:")
        print("- Strehl_ratio      = {:2.2f}".format(strehl_ratio))
        print("- MTF_ratio @ {:2.2f}  = {:2.2f}".format(lines_mm, mtf_50_ratio))
        print(
            "- freq @ {:2.2f}  real (lines/mm) = {:2.2f}".format(lines_mm, freq_50_real)
        )
        print(
            "- freq @ {:2.2f}  ideal (lines/mm) = {:2.2f}".format(
                lines_mm, freq_50_ideal
            )
        )

    return strehl_ratio, mtf_50_ratio, freq_50_real, freq_50_ideal


def gauss_spectrum(wavelengths, w_central, Dw, normalize=True):
    """
    returns weigths for a gaussian spectrum
    Parameters:
        wavelengths: array with wavelengths
        w_central: central wavelength
        Dw: width of the spectrum
        normalize: if True sum of weights is 1
    """

    weigths = np.exp(-((wavelengths - w_central) ** 2) / (2 * Dw**2))

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def lorentz_spectrum(wavelengths, w_central, Dw, normalize=True):
    """
    returns weigths for a gaussian spectrum
    Parameters:
        wavelengths: array with wavelengths
        w_central: central wavelength
        Dw: width of the spectrum
        normalize: if True sum of weights is 1
    """

    weigths = 1 / (1 + ((wavelengths - w_central) / (Dw / 2)) ** 2)

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def uniform_spectrum(wavelengths, normalize=True):
    """
    returns weigths for a gaussian spectrum
    Parameters:
        wavelengths: array with wavelengths
        w_central: central wavelength
        Dw: width of the spectrum
        normalize: if True sum of weights is 1
    """

    weigths = np.ones_like(wavelengths, dtype=float)

    if normalize is True:
        weights = weigths / weigths.sum()

    return weights


def normalize_field(self, new_field=False):
    """Normalize the field to maximum intensity.

    Parameters:
        (Scalar_field): field_*
        new_field (bool): If True returns a field, else returns a matrix
    Returns:
        (np.array): normalized field.
    """

    if self.type[0:6] == "Scalar":
        max_amplitude = np.sqrt(np.abs(self.u) ** 2).max()

        if new_field is False:
            self.u = self.u / max_amplitude
        else:
            field_new = self.duplicate()
            field_new.u = self.u / max_amplitude
            return field_new

    elif self.type[0:6] == "Vector":
        max_amplitude = np.sqrt(
            np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2
        ).max()

        if new_field is False:
            self.Ex = self.Ex / max_amplitude
            self.Ey = self.Ey / max_amplitude
            self.Ez = self.Ez / max_amplitude
        else:
            field_new = self.duplicate()
            field_new.Ex = self.Ex / max_amplitude
            field_new.Ey = self.Ey / max_amplitude
            field_new.Ez = self.Ez / max_amplitude
            return field_new


# def normalize(u, kind='intensity'):
#     """Normalizes a field to have intensity or amplitude, etc. 1

#     Parameters:
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

# def normalize_vector_deprecated(u):
#     """Normalizes a vector to have intensity or amplitude, etc. 1

#     Parameters:
#         u (numpy.array): vector (last dimension should have size 2 or 3)

#     Returns
#         u (numpy.array): normalized optical field
#     """
#     return u / np.linalg.norm(u)


def field_parameters(u, has_amplitude_sign=False):
    """Determines main parameters of field: amplitude intensity phase. All this parameters have the same dimension as u.

    Parameters:
        u (numpy.array): optical field (comes usually form field.u)
        has_amplitude_sign (bool): If True - amplitude = np.sign(u) * np.abs(u), Else: amplitude =  np.abs(u)

    Returns:
        amplitude (numpy.array): np.abs(u)
        intensity (numpy.array): np.abs(u)**2
        phase (numpy.array): np.angle(u)

    """

    intensity = np.abs(u) ** 2
    phase = np.angle(u)

    if has_amplitude_sign is True:
        amplitude = np.sign(np.real(u)) * np.abs(u)
    else:
        amplitude = np.abs(u)

    # amplitude = np.abs(u)
    # amplitude = u * np.exp(-1j * phase)
    amplitude = np.real(amplitude)

    return amplitude, intensity, phase


def convert_phase2heigths(phase, wavelength, n, n_background):
    """We have a phase and it is converted to a depth. It is useful to convert Scalar_mask_X to Scalar_mask_XZ

    phase(x,z)= k (n-n_0) h(x,z).

    Parameters:
        phase (np.array): Phases
        wavelength (float): wavelength
        n (float or complex): refraction index of material
        n_background (float): refraction index of background

    Returns:
        (np.array): depths related to phases
    """
    k = 2 * np.pi / wavelength
    n = np.real(n)

    return phase / (k * (n - n_background))


def convert_amplitude2heigths(amplitude, wavelength, kappa, n_background):
    """We have a phase and it is converted to a depth. It is useful to convert Scalar_mask_X to Scalar_mask_XZ.

    Parameters:
        phase (np.array): Phases
        wavelength (float): wavelength
        kappa (float): refraction index of material.
        n_background (float): refraction index of background

    Returns:
        (np.array): depths related to amplitudes
    """

    eps_depth = 1e-4

    amplitude[amplitude < eps_depth] = eps_depth

    depth = np.log(amplitude) * wavelength / (-2 * np.pi * kappa)

    return depth


def fresnel_equations_kx(
    kx,
    wavelength,
    n1,
    n2,
    outputs=[True, True, True, True],
    has_draw=True,
    kind="amplitude_phase",
):
    """Fresnel_equations where input are kx part of wavevector.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refraction index of first materia
        n2 (float): refraction index of second materia
        outputs (bool,bool,bool,bool): Selects the outputs to compute
        has_draw (bool, optional): if True, it draw. Defaults to False.
        kind (str): It draw 'amplitude_phase' or 'real_imag'

    Returns:
        _type_: t_TM, t_TE, r_TM, r_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength

    kz_1 = np.sqrt((n1 * k0) ** 2 - kx**2)

    alpha = (n2 * k0) ** 2 - kx**2
    normal = alpha >= 0
    reflexion_total = alpha < 0

    kz_2 = np.zeros_like(kx, dtype=complex)
    kz_2[normal] = np.sqrt(alpha[normal])
    kz_2[reflexion_total] = 1.0j * np.sqrt(-alpha[reflexion_total])

    t_TM, t_TE, r_TM, r_TE = None, None, None, None

    if outputs[0]:
        t_TM = 2 * n1 * n2 * kz_1 / (n2**2 * kz_1 + n1**2 * kz_2)  # paralelo
    if outputs[1]:
        t_TE = 2 * kz_1 / (kz_1 + kz_2)  # perpendicular
    if outputs[2]:
        r_TM = (n2**2 * kz_1 - n1**2 * kz_2) / (
            n2**2 * kz_1 + n1**2 * kz_2
        )  # paralelo
    if outputs[3]:
        r_TE = (kz_1 - kz_2) / (kz_1 + kz_2)  # perpendicular

    if has_draw and outputs.sum() > 0:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        if kind == "amplitude_phase":
            if outputs[0]:
                axs[0].plot(kx, np.abs(t_TM), "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(kx, np.abs(t_TE), "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(kx, np.abs(r_TM), "r-.", label="$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(kx, np.abs(r_TE), "b-.", label="$r_{\perp, TE}$")

            axs[0].legend()
            axs[0].grid()

            axs[1].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[0].set_title("amplitude")

            if outputs[0]:
                axs[1].plot(kx, np.angle(t_TM) / degrees, "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[1].plot(kx, np.angle(t_TE) / degrees, "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[1].plot(kx, np.angle(r_TM) / degrees, "r-.", label="$r_{\parallel, TM}$")
            if outputs[3]:
                axs[1].plot(kx, np.angle(r_TE) / degrees, "b-.", label="$r_{\perp, TE}$")

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[1].set_title(r"phase $(^o)$")
            axs[1].set_ylim(-190, 190)
            axs[1].set_yticks([-180, -90, 0, 90, 180])

        elif kind == "real_imag":
            if outputs[0]:
                axs[0].plot(kx, np.real(t_TM), "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(kx, np.real(t_TE), "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(kx, np.real(r_TM), "r-.", label="$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(kx, np.real(r_TE), "b-.", label="$r_{\perp, TE}$")

            axs[0].legend()
            axs[0].grid()

            axs[0].set_xlim(kx[0], kx[-1])
            axs[0].set_xlabel(r"$k_x$")
            axs[0].set_title("real")

            if outputs[0]:
                axs[1].plot(kx, np.imag(t_TM) / degrees, "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[1].plot(kx, np.imag(t_TE) / degrees, "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[1].plot(kx, np.imag(r_TM) / degrees, "r-.", label="$r_{\parallel, TM}$")
            if outputs[3]:
                axs[1].plot(kx, np.imag(r_TE) / degrees, "b-.", label="$r_{\perp, TE}$")

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(kx[0], kx[-1])
            axs[1].set_xlabel(r"$k_x$")
            axs[1].set_title(r"imag")

    # plt.xlim(theta[0]/degrees, theta[-1]/degrees)

    return t_TM, t_TE, r_TM, r_TE  # paralelo, perpendicular


def transmitances_reflectances_kx(
    kx, wavelength, n1, n2, outputs=[True, True, True, True], has_draw=False
):
    """Transmitances and reflectances, where input are kx part of wavevector.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refraction index of first materia
        n2 (float): refraction index of second materia
        has_draw (bool, optional): if True, it draw. Defaults to False.
        outputs (bool,bool,bool,bool): Selects the outputs to compute

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength

    kz_1 = np.sqrt((n1 * k0) ** 2 - kx**2)

    alpha = (n2 * k0) ** 2 - kx**2
    normal = alpha >= 0
    reflexion_total = alpha < 0

    kz_2 = np.zeros_like(kx, dtype=complex)
    kz_2[normal] = np.sqrt(alpha[normal])
    kz_2[reflexion_total] = 1.0j * np.sqrt(-alpha[reflexion_total])

    t_TM, t_TE, r_TM, r_TE = fresnel_equations_kx(
        kx, wavelength, n1, n2, outputs, has_draw=False
    )

    T_TM, T_TE, R_TM, R_TE = None, None, None, None

    if outputs[0]:
        T_TM = kz_2 / kz_1 * np.abs(t_TM) ** 2
    if outputs[1]:
        T_TE = kz_2 / kz_1 * np.abs(t_TE) ** 2
    if outputs[2]:
        R_TM = np.abs(r_TM) ** 2
    if outputs[3]:
        R_TE = np.abs(r_TE) ** 2

    if has_draw:
        plt.figure()
        if outputs[0]:
            plt.plot(kx, T_TM, "r", label="$T_{\parallel, TM}$")
        if outputs[1]:
            plt.plot(kx, T_TE, "b", label="$T_{\perp, TE}$")
        if outputs[2]:
            plt.plot(kx, R_TM, "r-.", label="$R_{\parallel, TM}$")
        if outputs[3]:
            plt.plot(kx, R_TE, "b-.", label="$R_{\perp, TE}$")

        plt.xlabel("$k_x$")
        plt.legend()
        plt.grid()

        # plt.xlim(theta[0]/degrees, theta[-1]/degrees)

    return T_TM, T_TE, R_TM, R_TE  # paralelo, perpendicular


def fresnel_equations(
    theta,
    wavelength,
    n1,
    n2,
    outputs=[True, True, True, True],
    has_draw=False,
    kind="amplitude_phase",
):
    """Fresnel equations and reflectances, where input are angles of incidence.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refraction index of first materia
        n2 (float): refraction index of second materia
        kind (str): It draw 'amplitude_phase' or 'real_imag'
        has_draw (bool, optional): if True, it draw. Defaults to False.
        kind (str): It draw 'amplitude_phase' or 'real_imag'

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """

    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength
    kx = n1 * k0 * np.sin(theta)

    t_TM, t_TE, r_TM, r_TE = fresnel_equations_kx(
        kx, wavelength, n1, n2, outputs, has_draw=False
    )

    if has_draw and outputs.sum() > 0:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        if kind == "amplitude_phase":
            if outputs[0]:
                axs[0].plot(theta / degrees, np.abs(t_TM), "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(theta / degrees, np.abs(t_TE), "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(theta / degrees, np.abs(r_TM), "r-.", label="$r_{\parallel, TM}$")
            if outputs[3]:
                axs[0].plot(
                    theta / degrees, np.abs(r_TE), "b-.", label="$r_{\perp, TE}$"
                )

            axs[0].legend()
            axs[0].grid()

            axs[0].set_xlim(theta[0] / degrees, theta[-1] / degrees)
            axs[0].set_xlabel(r"$\theta (^o)$")
            axs[0].set_title("amplitude")

            if outputs[0]:
                axs[1].plot(
                    theta / degrees,
                    np.angle(t_TM) / degrees,
                    "r",
                    label="$t_{\parallel, TM}$",
                )
            if outputs[1]:
                axs[1].plot(
                    theta / degrees,
                    np.angle(t_TE) / degrees,
                    "b",
                    label="$t_{\perp, TE}$",
                )
            if outputs[2]:
                axs[1].plot(
                    theta / degrees,
                    np.angle(r_TM) / degrees,
                    "r-.",
                    label="$r_{\parallel, TM}$",
                )
            if outputs[3]:
                axs[1].plot(
                    theta / degrees,
                    np.angle(r_TE) / degrees,
                    "b-.",
                    label="$r_{\perp, TE}$",
                )

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(theta[0] / degrees, theta[-1] / degrees)
            axs[1].set_xlabel(r"$\theta (^o)$")
            axs[1].set_title(r"phase $(^o)$")
            axs[1].set_ylim(-190, 190)
            axs[1].set_yticks([-180, -90, 0, 90, 180])

        elif kind == "real_imag":
            if outputs[0]:
                axs[0].plot(theta / degrees, np.real(t_TM), "r", label="$t_{\parallel, TM}$")
            if outputs[1]:
                axs[0].plot(theta / degrees, np.real(t_TE), "b", label="$t_{\perp, TE}$")
            if outputs[2]:
                axs[0].plot(
                    theta / degrees, np.real(r_TM), "r-.", label="$r_{\parallel, TM}$"
                )
            if outputs[3]:
                axs[0].plot(
                    theta / degrees, np.real(r_TE), "b-.", label="$r_{\perp, TE}$"
                )

            axs[0].legend()
            axs[0].grid()
            axs[0].set_xlabel(r"$\theta (^o)$")
            axs[0].set_xlim(theta[0] / degrees, theta[-1] / degrees)
            axs[0].set_title("real")

            if outputs[0]:
                axs[1].plot(
                    theta / degrees, np.imag(t_TM) / degrees, "r", label="$t_{\parallel, TM}$"
                )
            if outputs[1]:
                axs[1].plot(
                    theta / degrees,
                    np.imag(t_TE) / degrees,
                    "b",
                    label="$t_{\perp, TE}$",
                )
            if outputs[2]:
                axs[1].plot(
                    theta / degrees,
                    np.imag(r_TM) / degrees,
                    "r-.",
                    label="$r_{\parallel, TM}$",
                )
            if outputs[3]:
                axs[1].plot(
                    theta / degrees,
                    np.imag(r_TE) / degrees,
                    "b-.",
                    label="$r_{\perp, TE}$",
                )

            axs[1].legend()
            axs[1].grid()
            axs[1].set_xlim(theta[0] / degrees, theta[-1] / degrees)
            axs[1].set_xlabel(r"$\theta (^o)$")
            axs[1].set_title(r"imag")

    return t_TM, t_TE, r_TM, r_TE  # paralelo, perpendicular


def transmitances_reflectances(
    theta, wavelength, n1, n2, outputs=[True, True, True, True], has_draw=False
):
    """Transmitances and reflectances, where input are angles of incidence.

    Args:
        kx (np.array): kx
        wavelength (float): wavelength
        n1 (float): refraction index of first materia
        n2 (float): refraction index of second materia
        has_draw (bool, optional): if True, it draw. Defaults to False.
        outputs (bool,bool,bool,bool): Selects the outputs to compute

    Returns:
        _type_: T_TM, T_TE, R_TM, R_TE  (TM is parallel and TE is perpendicular)
    """
    outputs = np.array(outputs)

    k0 = 2 * np.pi / wavelength
    kx = k0 * n1 * np.sin(theta)

    T_TM, T_TE, R_TM, R_TE = transmitances_reflectances_kx(
        kx, wavelength, n1, n2, outputs, has_draw=False
    )

    if has_draw:
        plt.figure()
        if outputs[0]:
            plt.plot(theta / degrees, T_TM, "r", label="$T_{\parallel, TM}$")
        if outputs[1]:
            plt.plot(theta / degrees, T_TE, "b", label="$T_{\perp, TE}$")
        if outputs[2]:
            plt.plot(theta / degrees, R_TM, "r-.", label="$R_{\parallel, TM}$")
        if outputs[3]:
            plt.plot(theta / degrees, R_TE, "b-.", label="$R_{\perp, TE}$")

        plt.xlim(theta[0] / degrees, theta[-1] / degrees)
        plt.xlabel(r"$\theta (^o)$")
        plt.legend()
        plt.grid()


    return T_TM, T_TE, R_TM, R_TE  # paralelo, perpendicular

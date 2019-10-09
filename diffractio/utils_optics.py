# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" General purpose optics functions """

import pandas as pd
from numpy import (angle, arcsin, cos, exp, imag, meshgrid, pi, real, sign,
                   sin, sqrt, unwrap)

from . import degrees, np, plt
from .utils_math import fft_convolution1d, fft_convolution2d, nearest


def roughness_1D(x, t, s, kind='normal'):
    """Rough surface, 1D

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.

    Parameters:
        x (numpy.array): array with x positions
        t (float): correlation lens
        s (float): std of roughness
        kind (str): 'normal', 'uniform'

    Returns:
        (numpy.array) Topography of roughnness in microns.
    """

    ancho = x[-1] - x[0]
    dx = x[1] - x[0]

    # Surface parameters

    L_ancho = ancho / (2 * dx)
    M = round(4 * t / (np.sqrt(2) * dx))

    N_ancho = int(np.floor(L_ancho + M))

    # GENERACION DE LA SUPERFICIE
    # Desplazamientos para correlacionar
    desp_ancho = np.arange(-M, M + 1)

    # (dominio del kernel de la correlacion)
    desp_ancho = desp_ancho * dx
    pesos = np.exp(-2 * (desp_ancho**2 / t**2))

    # print np.sqrt((pesos**2).sum())
    pesos = np.abs(pesos / np.sqrt(
        (pesos**2).sum()))  # p' q' la suma de sus squares de 1

    # Alturas no correlacionadas # con distribucion N(0,s)
    if kind == 'normal':
        h_no_corr = s * np.random.randn(2 * N_ancho + 1)
        h_corr = fft_convolution1d(h_no_corr, pesos)
        h_corr = h_corr[0:len(x)]
    elif kind == 'uniform':
        h_corr = s * (np.random.rand(len(x)) - 0.5)
    return h_corr


def roughness_2D(x, y, t, s):
    """Rough surface, 2D

    References:
        JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger p.224.

    Parameters:
        x (numpy.array): x positions
        y (numpy.array): y positions
        t (float, float): (tx, ty), correlation length of roughness
        s (float): std of heights

    Returns:
        (numpy.array) Topography of roughnness in microns.

    Example:
        roughness(t=(50 * um, 25 * um), s=1 * um)
    """

    if isinstance(t, (float, int, complex)):
        t = (t, t)

    tx, ty = t

    ancho = x[-1] - x[0]
    largo = y[-1] - y[0]
    dx = x[1] - x[0]
    # PARAMETROS DE LA SUPERFICIE
    L_ancho = ancho / (2 * dx)
    L_largo = largo / (2 * dx)
    M = round(4 * tx / (sqrt(2) * dx))
    N_ancho = int(np.floor(L_ancho + M))
    N_largo = int(np.floor(L_largo + M))

    # GENERACION DE LA SUPERFICIE
    # Desplazamientos para correlacionar
    desp_ancho, desp_largo = meshgrid(
        np.arange(-M, M + 1), np.arange(-M, M + 1))
    # (dominio del kernel de la correlacion)
    desp_ancho = desp_ancho * dx
    desp_largo = desp_largo * dx

    pesos = exp(-2 * (desp_ancho**2 / tx**2 + desp_largo**2 / ty**2))
    # print sqrt((pesos**2).sum())
    pesos = np.abs(pesos / sqrt((pesos**2).sum()))
    # p' q' la suma de sus squares de 1

    # Alturas no correlacionadas # con distribucion N(0,s)
    h_no_corr = s * np.random.randn(2 * N_ancho + 1, 2 * N_largo + 1)
    # Alturas correlacionadas en el plano:
    # Las heights salen de correlacionar entre si las h no corr
    # reemplazando una de ellas por el promedio ponderado con sus vecinas

    h_corr = fft_convolution2d(h_no_corr, pesos)

    # En en la conv 2D los pesos deben ir 2 dos en el argumentos
    h_corr = h_corr[0:len(x), 0:len(y)]
    return h_corr


def beam_width_1D(u, x):
    """One dimensional beam width, according to D4σ or second moment width.
    References:
        https://en.wikipedia.org/wiki/Beam_diameter

    Parameters:
        u (np.array): field (not intensity).
        x (np.array): x

    Returns:
        (float): width
        (float): x_mean
    """

    intensity = np.abs(u)**4

    P = (intensity).sum()

    x_mean = (intensity * x).sum() / P
    x2_mean = (intensity * (x - x_mean)**2).sum() / P

    width_x = 4 * sqrt(x2_mean)

    return width_x, x_mean


def width_percentaje(x, y, percentaje=0.5, verbose=False):
    """ beam width (2*sigma) given at a certain height from maximum

    y=np.exp(-x**2/(s**2))  percentaje=1/e -> width = 2*s
    y=np.exp(-x**2/(s**2))  percentaje=1/e**4 -> width = 4*s
    y=np.exp(-x**2/(2*s**2))  percentaje=1/e**2 =  -> width = 4*s

    Parameters:
        x (np.array): x
        y (np.array): y
        percentaje (float): percentaje of height. For example: 0.5

    Returns:
        (float): width, width of at given %
        (list): x_list: (x[i_left], x[i_max], x[i_right])
        (list): x_list: (i_left, i_max, i_right)
    """

    maximum = y.max()
    level = percentaje * maximum
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


def beam_width_2D(x, y, intensity):
    """2D beam width
    ISO11146 width
    References:
        * https://en.wikipedia.org/wiki/Beam_diameter

        * http://www.auniontech.com/ueditor/file/20170921/1505982360689799.pdf

    Parameters:
        x (np.array): 1d x
        y (np.array): 1d y
        intensity (np.array):  intensity

    Returns:
        (float): dx width x
        (float): dy width y
        (float): principal_axis, angle
        (str): (x_mean, y_mean, x2_mean, y2_mean, xy_mean), Moments

    """
    X, Y = np.meshgrid(x, y)
    intensity = intensity - intensity.min()

    P = intensity.sum()
    x_mean = (intensity * X).sum() / P
    y_mean = (intensity * Y).sum() / P
    x2_mean = (intensity * (X - x_mean)**2).sum() / P
    y2_mean = (intensity * (Y - y_mean)**2).sum() / P
    xy_mean = (intensity * (X - x_mean) * (Y - y_mean)).sum() / P
    gamma = (x2_mean - y2_mean) / np.abs(x2_mean - y2_mean + 1e-16)
    principal_axis = 0.5 * np.arctan2(2 * xy_mean, x2_mean - y2_mean)
    dx = 2 * sqrt(2) * sqrt(x2_mean + y2_mean + gamma * sqrt(
        (x2_mean - y2_mean)**2 + 4 * xy_mean**2))
    dy = 2 * sqrt(2) * sqrt(x2_mean + y2_mean - gamma * sqrt(
        (x2_mean - y2_mean)**2 + 4 * xy_mean**2))

    return dx, dy, principal_axis, (x_mean, y_mean, x2_mean, y2_mean, xy_mean)


def refraction_index(filename, wavelength, raw=False, has_draw=True):
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

    wavelengths = data['Wavelength, µm'].values.astype(float)
    n = data['n'].values.astype(float)
    kappa = data['k'].values.astype(float)

    if has_draw is True:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('wavelengths (nm)')
        ax1.plot(wavelengths, n, 'r', label='n')
        ax1.set_ylabel('n', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        ax2 = ax1.twinx()
        ax2.plot(wavelengths, kappa, 'b', label=r'$\kappa$')
        ax2.set_ylabel(r'$\kappa$', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
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


def FWHM1D(x,
           intensity,
           percentaje=0.5,
           remove_background='mean',
           has_drawing=True):
    """FWHM

    remove_background = 'min', 'mean', 'None'"""

    if remove_background == 'mean':
        I_background = intensity.mean()
    elif remove_background == 'min':
        I_background = intensity.min()
    else:
        I_background = np.zeros_like(intensity)

    intensity = intensity - I_background

    delta_x = x[1] - x[0]
    amp_max = intensity.max()
    amp_med = amp_max * percentaje
    i_max = np.where(intensity == amp_max)
    i_max = int(i_max[0][0])
    left = intensity[0:i_max]
    right = intensity[i_max::]

    i_left, _, distance_left = nearest(left, percentaje * amp_max)
    slope_left = (intensity[i_left + 1] - intensity[i_left]) / delta_x

    i_right, _, distance_right = nearest(right, percentaje * amp_max)
    slope_right = (
        intensity[i_max + i_right] - intensity[i_max + i_right - 1]) / delta_x

    i_right = i_right + i_max

    x_right = i_right * delta_x - distance_right / slope_right
    x_left = i_left * delta_x - distance_left / slope_left

    FWHM_x = x_right - x_left

    amp_max = amp_max + I_background
    amp_med = amp_med + I_background

    intensity = intensity + I_background

    if has_drawing is True:
        if remove_background is True:
            intensity = intensity + intensity.min()
        plt.figure()

        plt.plot(x, intensity, 'k', lw=2)
        plt.plot([x[0], x[-1]], [amp_max, amp_max], 'r--')
        plt.plot([x[0], x[-1]], [amp_med, amp_med], 'r--')

        plt.plot(x[i_max], intensity[i_max], 'ro', ms=8)
        plt.plot(
            x[int(i_left)] + distance_left, intensity[int(i_left)], 'ro', ms=8)
        plt.plot(
            x[int(i_right)] + distance_right,
            intensity[int(i_right)],
            'ro',
            ms=8)

    return FWHM_x


def FWHM2D(x,
           y,
           intensity,
           percentaje=0.5,
           remove_background='mean',
           has_drawing=True):

    Ix = intensity.sum(axis=0)
    Iy = intensity.sum(axis=1)

    FWHM_x = FWHM1D(
        x, Ix, percentaje, remove_background, has_drawing=has_drawing)
    FWHM_y = FWHM1D(
        y, Iy, percentaje, remove_background, has_drawing=has_drawing)

    return FWHM_x, FWHM_y


def detect_intensity_range(x,
                           intensity,
                           percentage=0.95,
                           has_draw=True,
                           logarithm=True):
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
        fig, ax = plt.subplots()

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

        ax.plot(x, I2, c='r', alpha=1, lw=4)

        x_bordes = [x_min, x_max, x_max, x_min, x_min]
        y_bordes = [I_min2, I_min2, I_max2, I_max2, I_min2]

        ax.fill(x_bordes, y_bordes, c='r', alpha=0.25)

    return x_min, x_max


def MTF_ideal(frequencies,
              wavelength,
              diameter,
              focal,
              kind,
              verbose=False,
              has_draw=False):
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
    frequency_max = 1000. / (wavelength * F_number)  # porque mido en micras
    fx_norm = np.abs(frequencies / frequency_max)

    if kind == '1D':
        MTF = 1 - np.abs(fx_norm)
        MTF[fx_norm > 1] = 0

    elif kind == '2D':
        fx2 = np.arccos(fx_norm)
        MTF = np.real(2 / np.pi * (fx2 - np.cos(fx2) * np.sin(fx2)))

        # otra definición: https://www.optikos.com/wp-content/uploads/2015/10/How-to-Measure-MTF-and-other-Properties-of-Lenses.pdf
        # MTF = np.real(2/np.pi*(np.arccos(fx_norm)-fx_norm*np.sqrt(1-fx_norm**2)))

        # isH1 = MTF > 1
        # MTF[isH1] = 2 - MTF[isH1]

    if verbose is True:
        print("frecuencia de corte = {:4.2f} lineas/mm".format(frequency_max))

    if has_draw is True:
        plt.figure()
        plt.plot(frequencies, MTF, 'k')
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
        print("- MTF_ratio @ {:2.2f}  = {:2.2f}".format(
            lines_mm, mtf_50_ratio))
        print("- freq @ {:2.2f}  real (lines/mm) = {:2.2f}".format(
            lines_mm, freq_50_real))
        print("- freq @ {:2.2f}  ideal (lines/mm) = {:2.2f}".format(
            lines_mm, freq_50_ideal))

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

    weigths = exp(-(wavelengths - w_central)**2 / (2 * Dw**2))

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

    weigths = 1 / (1 + ((wavelengths - w_central) / (Dw / 2))**2)

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


def normalize(u, kind='intensity'):
    """Normalizes a field to have intensity or amplitude, etc. 1

    Parameters:
        u (numpy.array): optical field (comes usually form field.u)
        kind (str): 'intensity, 'amplitude', 'logarithm'... other.. Normalization technique

    Returns
        u (numpy.array): normalized optical field
    """

    if kind == 'intensity':
        intensity_max = (np.abs(u)).max()
        u = u / intensity_max
    elif kind == 'amplitude':
        amplitude_max = np.sqrt(np.abs(u)).max()
        u = u / amplitude_max
    if kind == 'logarithm':
        log_max = (np.log(np.sqrt(np.abs(u)))).max()
        u = u / log_max

    return u


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

    intensity = np.abs(u)**2
    phase = np.angle(u)
    if has_amplitude_sign is True:
        amplitude = np.sign(u) * np.abs(u)
    else:
        amplitude = np.abs(u)

    # amplitude = np.abs(u)
    # amplitude = u * np.exp(-1j * phase)
    amplitude = np.real(amplitude)

    return amplitude, intensity, phase


def fresnel_coefficients_dielectric(theta_i, n1, n2):
    """Components rs, rp, ts y tp - Fresnel queations

    Parameters:
        theta_i (numpy.array or float): angle of incidence.
        n1 (float): refraction index of first medium
        n2 (float): refraction index of second medium

    Returns:
        (numpy.array or float): r_perp
        (numpy.array or float): r_par
        (numpy.array or float): t_perp
        (numpy.array or float): t_par
    """

    theta_t = arcsin(n1 * sin(theta_i) / n2)

    cos_i = cos(theta_i)
    cos_t = cos(theta_t)

    r_par = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    r_perp = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    t_par = (2 * n1 * cos_i) / (n2 * cos_i + n1 * cos_t)
    t_perp = (2 * n1 * cos_i) / (n1 * cos_i + n2 * cos_t)

    return r_perp, r_par, t_perp, t_par


def reflectance_transmitance_dielectric(theta_i, n1, n2):
    """Transmittances R_perp, R_par, T_perp, T_par - Fresnel queations

    Parameters:
        theta_i (numpy.array or float): angle of incidence.
        n1 (float): refraction index of first medium
        n2 (float): refraction index of second medium

    Returns:
        (numpy.array or float): R_perp
        (numpy.array or float): R_par
        (numpy.array or float): T_perp
        (numpy.array or float): T_par
    """
    r_perp, r_par, t_perp, t_par = fresnel_coefficients_dielectric(
        n1, theta_i, n2)

    theta_t = arcsin(n1 * sin(theta_i) / n2)

    cos_i = cos(theta_i)
    cos_t = cos(theta_t)

    R_perp = np.abs(r_perp)**2
    R_par = np.abs(r_par)**2
    T_perp = np.abs(t_perp)**2 * (n2 * cos_t) / (n1 * cos_i)
    T_par = np.abs(t_par)**2 * (n2 * cos_t) / (n1 * cos_i)

    return R_perp, R_par, T_perp, T_par


def fresnel_coefficients_complex(theta_i, n1, n2c):
    """
    Calcula las components rs y rp mediante las eq. de Fresnel
    n^=n-ik
    example:
        theta_i=linspace(0*degrees,90*degrees,10),

    Los parametros de entrada pueden ser arrays de numeros.
    Para drawlos el array debe ser theta_i
    n2c puede ser complejo
    """
    # Precalculos
    kiz = cos(theta_i)
    ktcz = sqrt(n2c**2 - n1**2 * sin(theta_i)**2)
    ktc2 = n2c**2
    ki2 = n1**2

    # Calculo de los coeficientes de Fresnel
    r_perp = (kiz - ktcz) / (kiz + ktcz)
    t_perp = 2 * kiz / (kiz + ktcz)
    r_par = (kiz * ktc2 - ktcz * ki2) / (kiz * ktc2 + ktcz * ki2)
    t_par = 2 * kiz * ktc2 / (kiz * ktc2 + ktcz * ki2)

    return r_perp, r_par, t_perp, t_par


def reflectance_transmitance_complex(theta_i, n1, n2c):
    """
    Calcula las components rs y rp mediante las eq. de Fresnel
    n^=n-ik
    example:
        theta_i=linspace(0*degrees,90*degrees,10),

    Los parametros de entrada pueden ser arrays de numeros.
    Para drawlos el array debe ser theta_i
    n2c puede ser complejo
    """

    # Coeficientes de Fresnel
    r_perp, r_par, t_perp, t_par = fresnel_coefficients_complex(
        n1, theta_i, n2c)

    # Reflectancia
    R_perp = np.abs(r_perp)**2
    R_par = np.abs(r_par)**2

    # Precalculo
    kiz = cos(theta_i)
    ki2 = n1**2
    ktcz = sqrt(n2c**2 - n1**2 * sin(theta_i)**2)
    ktc2 = n2c**2
    n2R = real(n2c)
    kappa2 = imag(n2c)
    B = n2R**2 - kappa2**2 - n1**2 * sin(theta_i)**2
    ktz = sqrt(0.5 * (B + sqrt(B**2 + 4 * n2R**2 * kappa2**2)))

    # Transmitancias
    T_perp = ktz * np.abs(t_perp)**2 / kiz
    T_par = ki2 * real(ktcz / ktc2) * np.abs(t_par)**2 / kiz

    return R_perp, R_par, T_perp, T_par


def draw_fresnel_coefficients(theta_i,
                              n1,
                              n2,
                              r_perp,
                              r_par,
                              t_perp,
                              t_par,
                              filename=''):
    """
    Dibuja las ecuaciones de fresnel en function del angle de entrada
    """

    # Generacion de la figura
    plt.figure()

    # Amplitud
    plt.subplot(1, 2, 1)
    plt.plot(
        theta_i / degrees,
        np.abs(r_perp) * sign(r_perp),
        'k--',
        lw=2,
        label=u"$r_{\perp}$")
    plt.plot(
        theta_i / degrees,
        np.abs(r_par) * sign(r_par),
        'k',
        lw=2,
        label=u"$r_{\parallel}$")
    plt.plot(
        theta_i / degrees,
        np.abs(t_perp) * sign(t_perp),
        'r--',
        lw=2,
        label=r"$t_{\perp}$")
    plt.plot(
        theta_i / degrees,
        np.abs(t_par) * sign(t_par),
        'r',
        lw=2,
        label=r"$t_{\parallel}$")

    # Leyenda de los ejes
    plt.xlabel(r"$\phi (degrees)$", fontsize=22)
    plt.ylabel(r"$Amplitud$", fontsize=22)
    plt.legend(loc=2, prop={'size': 18})

    # Fase
    plt.subplot(1, 2, 2)
    plt.plot(
        theta_i / degrees,
        unwrap(angle(r_perp), 2 * pi),
        'k--',
        lw=2,
        label=r"$r_{\perp}$")
    plt.plot(
        theta_i / degrees,
        unwrap(angle(r_par), 2 * pi),
        'k',
        lw=2,
        label=r"$r_{\parallel}$")
    plt.plot(
        theta_i / degrees,
        unwrap(angle(t_perp), 2 * pi),
        'r--',
        lw=2,
        label=r"$t_{\perp}$")
    plt.plot(
        theta_i / degrees,
        unwrap(angle(t_par), 2 * pi),
        'r',
        lw=2,
        label=r"$t_{\parallel}$")

    # Leyenda de los ejes
    plt.xlabel(r"$\phi (degrees)$", fontsize=22)
    plt.ylabel(r"$phase$", fontsize=22)
    plt.legend(loc=2, prop={'size': 18})

    if not filename == '':
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)


def drawTransmitancias(theta_i,
                       n1,
                       n2,
                       R_perp,
                       R_par,
                       T_perp,
                       T_par,
                       filename=''):
    """
    Dibuja las ecuaciones de fresnel en function del angle de entrada
    """
    # Generacion de la figura
    plt.figure()
    # drawing
    plt.subplot(1, 1, 1)
    plt.plot(
        theta_i / degrees, np.real(R_perp), 'k--', lw=2, label=u"$R_{\perp}$")
    plt.plot(
        theta_i / degrees, np.real(R_par), 'k', lw=2, label=u"$R_{\parallel}$")
    plt.xlabel(r"$\phi (degrees)$", fontsize=22)
    plt.ylabel(r"$Amplitud$", fontsize=22)
    plt.ylim(-0.01, 1.01)
    plt.plot(
        theta_i / degrees, np.real(T_perp), 'r--', lw=2, label=r"$T_{\perp}$")
    plt.plot(
        theta_i / degrees, np.real(T_par), 'r', lw=2, label=r"$T_{\parallel}$")
    plt.xlabel(r"$\phi (degrees)$", fontsize=22)
    plt.ylabel(r"$intensity$", fontsize=22)
    plt.ylim(-0.1, 2)
    plt.legend(loc=3, prop={'size': 18})

    if not filename == '':
        plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)

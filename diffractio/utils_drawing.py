# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for drawing """

import os

import matplotlib.animation as manimation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from . import degrees, eps, mm, params_drawing
from .utils_optics import field_parameters

percentaje_intensity = params_drawing['percentaje_intensity']


def view_image(filename):
    """reproduces image

    Parameters:
        filename (str): filename
    """
    if not filename == '' and filename is not None:
        mpimg.imread(filename)
        plt.show()


def concatenate_drawings(kind1='png',
                         kind2='png',
                         nx=5,
                         ny=3,
                         geometria_x=256,
                         geometria_y=256,
                         raiz='fig4_nsensors_1',
                         nombreFigura="figura2.png",
                         directorio=""):
    listaArchivos = os.listdir(directorio)
    print(listaArchivos)

    texto_ficheros = ""
    for fichero in sorted(listaArchivos):
        if fichero[-3:] == kind1 and fichero[0:len(raiz)] == raiz:
            print(fichero)
            texto_ficheros = texto_ficheros + " " + directorio + fichero

    os.system("cd " + directorio)
    texto1 = "montage %s -tile %dx%d -geometry %d x %d -5-5 %s" % (
        texto_ficheros, nx, ny, geometria_x, geometria_y, nombreFigura)

    print(texto1)
    os.system(texto1)

    print("Finished")


def draw2D(
        image,
        x,
        y,
        xlabel="$x  (\mu m)$",
        ylabel="$y  (\mu m)$",
        title="",
        color="YlGnBu",  # YlGnBu  seismic
        interpolation='nearest',  # 'bilinear', 'nearest'
        scale='scaled',
        reduce_matrix='standard',
        range_scale='um',
        verbose=False):
    """makes a drawing of XY

    Parameters:
        image (numpy.array): image to draw
        x (numpy.array): positions x
        y (numpy.array): positions y
        xlabel (str): label for x
        ytlabel (str): label for y
        title (str): title
        color (str): color
        interpolation (str): 'bilinear', 'nearest'
        scale (str): kind of axis (None, 'equal', 'scaled', etc.)
        range_scale (str): 'um' o 'mm'
        verbose (bool): if True prints information

    Returns:
        id_fig: handle of figure
        IDax: handle of axis
        IDimage: handle of image
    """
    if reduce_matrix in (None, '', []):
        pass
    elif reduce_matrix == 'standard':
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_y / 500)

        if reduction_x == 0:
            reduction_x = 1
        if reduction_y == 0:
            reduction_y = 1

        image = image[::reduction_x, ::reduction_y]
    else:
        image = image[::reduce_matrix[0], ::reduce_matrix[1]]

    if verbose is True:
        print(("image size {}".format(image.shape)))

    id_fig = plt.figure()
    IDax = id_fig.add_subplot(111)

    if range_scale == 'um':
        extension = (x[0], x[-1], y[0], y[-1])
    else:
        extension = (x[0] / mm, x[-1] / mm, y[0] / mm, y[-1] / mm)
        xlabel = "x (mm)"
        ylabel = "y (mm)"

    IDimage = plt.imshow(
        image,
        interpolation=interpolation,
        aspect='auto',
        origin='lower',
        extent=extension)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.axis(extension)
    if scale not in ('', None, []):
        plt.axis(scale)
    IDimage.set_cmap(color)
    plt.tight_layout()
    return id_fig, IDax, IDimage


# def draw3D(image, x, y, xlabel="", ylabel="", color="YlGnBu"):
#     X, Y = np.meshgrid(x, y)
#     mlab.figure(fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
#     mlab.mesh(X, Y, image, colormap=color)
#     mlab.xlabel(xlabel)
#     mlab.ylabel(ylabel)
#     # mlab.view(.0, -5.0, 4)


def draw_several_fields(fields,
                        titles='',
                        title='',
                        figsize='',
                        kinds='',
                        logarithm=False,
                        normalize=False):
    """Draws several fields in subplots

    Parameters:
        fields (list): list with several scalar_fields_XY
        titles (list): list with titles
        title (str): suptitle
        kinds (list): list with kinds of figures (amplitude', 'intensity', 'phase', 'real_field', 'contour')
        logarithm (bool): If True, intensity is scaled in logarithm
        normalize (bool): If True, max(intensity)=1
    """

    orden = [[1, 1], [2, 1], [3, 1], [2, 2], [3, 2], [3, 2]]
    length = [(10, 8), (10, 5), (11, 5), (9, 7), (12, 9), (12, 9)]

    num_dibujos = len(fields)
    fil = orden[num_dibujos - 1][0]
    col = orden[num_dibujos - 1][1]

    if figsize == '':
        figsize = length[num_dibujos - 1]

    id_fig = plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    num_dibujos = len(fields)

    percentaje_intensity = params_drawing['percentaje_intensity']

    for i in sorted(range(num_dibujos)):
        c = fields[i]
        id_fig.add_subplot(col, fil, i + 1)
        extension = [c.x.min(), c.x.max(), c.y.min(), c.y.max()]
        amplitude, intensity, phase = field_parameters(
            c.u, has_amplitude_sign=True)

        if kinds == '':
            image = intensity
            colormap = params_drawing['color_intensity']
            kind = 'intensity'
        else:
            kind = kinds[i]
            if kinds[i] == 'intensity':
                image = intensity
                colormap = params_drawing['color_intensity']
            elif kinds[i] == 'phase':
                phase = phase / degrees
                phase[intensity < percentaje_intensity * (intensity.max())] = 0

                colormap = params_drawing['color_phase']
                image = phase
            elif kinds[i] == 'amplitude':
                image = amplitude
                colormap = params_drawing['color_amplitude']
            elif kinds[i] == 'real':
                percentaje_intensity = params_drawing['percentaje_intensity']
                rf = np.real(c.u)
                intensity = np.abs(c.u)**2
                rf[intensity < percentaje_intensity * (intensity.max())] = 0

                image = np.real(c.u)
                colormap = params_drawing['color_real']

            if logarithm is True and kinds[i] in ('intensity', 'amplitude', 'real'):
                image = np.log(image + 1)

            if normalize == 'maximum' and kinds[i] in ('intensity', 'amplitude', 'real'):
                image = image / image.max()

        plt.imshow(
            image,
            interpolation='bilinear',
            aspect='auto',
            origin='lower',
            extent=extension)

        plt.set_cmap(colormap)
        if titles != '':
            plt.title(titles[i], fontsize=24)

        plt.suptitle(title, fontsize=26)
        plt.axis('scaled')
        plt.axis(extension)
        plt.colorbar(orientation='horizontal', shrink=0.5)
        if kind == 'phase':
            plt.clim(-180, 180)


def change_image_size(image_name,
                      length='800x600',
                      final_filename='prueba.png',
                      dpi=300):
    """change the size with imageMagick

        Parameters:
            image_name (str): name of file
            length (str): size of image
            final_filename (str): final filename
            dpi (int): dpi

        Examples:

        convert image_name -resize '1000' -units 300 final_filename.png
            - anchura 1000 - mantiene forma
        convert image_name -resize 'x200' final_filename.png
            - height  200  - mantiene forma
        convert image_name -resize '100x200>' final_filename.png
            - mantiene forma, lo que sea mayor
        convert image_name -resize '100x200<' final_filename.png
            - mantiene forma, lo que sea menor
        convert image_name -resize '@1000000' final_filename.png
            - mantiene la forma, con 1Mpixel
        convert image_name -resize '100x200!' final_filename.png
            - obliga a tener el tamaño, no mantiene escala
        """
    texto = "convert {} -resize {} {}".format(image_name, length,
                                              final_filename)
    print(texto)
    os.system(texto)


def extract_image_from_video(nombre_video=None,
                             num_frame="[0, ]",
                             final_filename='prueba.png'):
    """Extract images form a video using imageMagick.

    convert 'animacion.avi[15,]' animacion_frame.png. Extracts frame 15 (ony 15)
    convert 'animacion.avi[15]' animacion_frame.png. Extracts the first 15
    convert 'animacion.avi[5,10]' animacion_frame.png. Extracts frame 5 and 10
    """

    texto = "convert '%s%s' %s" % (nombre_video, num_frame, final_filename)
    print(texto)
    os.system(texto)


def normalize_draw(u, logarithm=0, normalize=False, cut_value=None):
    """
    Gets a field and changes its caracteristics for drawing

    Parameters:
        u (field): field
        logarithm (float): logarithm to image: np.log(logarithm*u + 1)
        normalize (str or bool): False, 'mean', 'intensity'
    """

    u = np.real(u)
    # u[u < 0] = 0

    if logarithm > 0:
        u = np.log(logarithm * u + 1)

    if normalize is False:
        pass
    elif normalize == 'maximum':
        u = u / (np.abs(u).max() + eps)
    elif normalize == 'mean':
        u = u / u.mean()

    if cut_value not in ([], '', 0, None):
        u[u > cut_value] = cut_value

    return u


def prepare_drawing(u, kind='intensity', logarithm=False, normalize=False):
    """It is necessary that figure is previously defined: plt.figure()

    Parameters:
        u - field
        kind - 'intensity', 'amplitude', 'phase'
        logarithm - True or False
        normalize: False, 'maximum', 'intensity', 'area'

    Returns:
        returns (numpy.array): I_drawing for direct plotting
    """
    amplitude, intensity, phase = field_parameters(u)

    if kind == 'intensity':
        I_drawing = intensity
        I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        # plt.title('Intensity')
    elif kind == 'amplitude':
        I_drawing = amplitude
        I_drawing = normalize_draw(I_drawing, logarithm, normalize)
        # plt.title('Amplitude')
    elif kind == 'phase':
        I_drawing = phase
        # plt.title('phase')
    else:
        print("bad kind parameter")
        return None
    return I_drawing


def prepare_video(fps=15, title='', artist='', comment=''):
    FFMpegWriter = manimation.writers['ffmpeg']  # ffmpeg mencoder
    metadata = dict(title=title, artist='artist', comment='comment')
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    return writer


def make_video_from_file(self, files, filename=''):
    print("Start", files)
    if not (filename) == '':
        print('Making movie animation.mpg - this make take a while')
        texto = "mencoder 'mf://_tmp*.png' -mf kind=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " + filename
        # texto = "mencoder 'mf://home/_tmp*.png' -mf kind=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o " + filename
        os.system(texto)
        # os.system("convert _tmp*.png animation2.gif")  # este sale muy grande
        # esto podria hacer mas pequeno convert -geometry 400 -loop 5  animation2.gif animation3.gif
        # cleanup
        print(files)
        for fname in files:
            os.remove(fname)
    print("exit", files)


def reduce_matrix_size(reduce_matrix, x, y, image, verbose=False):
    """Reduces the size of matrix for drawing purposes. If the matrix is very big, the drawing process is slow.

    Parameters:
        reduce_matrix (str or (int, int)): if str: 'standard', if (int, int) reduction_factor.
        x (np.array): array with x.
        y (np.array): array with y or z
        image (np.array): image to reduce the size.
        verbose (bool): if True, prints info

    Returns:
        (np.array): reduced image
    """
    image_ini = image.shape
    if reduce_matrix in (None, '', []):
        pass
    elif reduce_matrix == 'standard':
        num_x = len(x)
        num_y = len(y)
        reduction_x = int(num_x / 500)
        reduction_y = int(num_y / 500)

        if reduction_x > 2 and reduction_y > 2:
            image = image[::reduction_x, ::reduction_y]
            # print("reduction = {}, {}".format(reduction_x, reduction_y))
        else:
            pass
    else:
        image = image[::reduce_matrix[0], ::reduce_matrix[1]]

    if verbose:
        print(("reduce_matrix_size: size ini = {}, size_final = {}".format(
            image_ini, image.shape)))
    return image

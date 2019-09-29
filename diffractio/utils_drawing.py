# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for drawing """

import os

import matplotlib.animation as manimation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from . import eps, mm
from .utils_optics import field_parameters

#
# def get_screen_sizes():
#     """
#     screens[i].width,  screens[i].height, screens[i].x, screens[i].y
#     """
#     platform = pyglet.window.get_platform()
#     display = platform.get_default_display()
#
#     screens = []
#     for screen in display.get_screens():
#         screens.append(screen)
#
#     for i in range(len(screens)):
#         print(screens[i].width, screens[i].height, screens[i].x, screens[i].y)
#
#     return screens

#
# def send_image_to_screen(image1, size, position):
#     """This is a small script to demonstrate using Tk to show PIL Image objects.
#     The advantage of this over using Image.show() is that it will reuse the
#     same window, so you can show multiple images without opening a new
#     window for each image.
#
#     This will simply go through each file in the current directory and
#     try to display it. If the file is not an image then it will be skipped.
#     Click on the image display window to go to the next image.
#
#     Noah Spurrier 2007
#     """
#
#     root = Tkinter.Tk()
#     # root.geometry('+%d+%d' % (100,100))
#     w, h = size
#     px, py = position
#     root.overrideredirect(1)
#     root.geometry("{:d}x{:d}+{:d}+{:d}".format(w, h, px, py))
#     # root.geometry("%dx%d+0+0" % (w, h))
#
#     root.focus_set()  # <-- move focus to this widget
#
#     print("w={}, h={}".format(image1.size[0], image1.size[1]))
#     root.geometry('%dx%d' % (image1.size[0], image1.size[1]))
#     tkpi = ImageTk.PhotoImage(image1)
#     # label_image = Tkinter.Label(root, image=tkpi)
#     # label_image.place(x=0, y=0, width=image.size[0], height=image.size[1])
#     #
#     # root.title(f)
#     # if old_label_image is not None:
#     #     old_label_image.destroy()
#     # old_label_image = label_image


def get_image():
    """TODO: lost function"""

    kinds = ('png', 'jpg', 'gif', 'tif', 'xpm')
    filename = select_file(kinds, nombreGenerico='images')
    return filename


def view_image(filename):
    """reproduces image

    Parameters:
        filename (str): filename
    """
    if not filename == '' and filename is not None:
        mpimg.imread(filename)
        plt.show()


def convert_drawings(kind1='jpg',
                     kind2='png',
                     densidad=300,
                     escala=100,
                     quality=100,
                     directorio="/home/luismiguel/Escritorio/prueba"):

    listaArchivos = os.listdir(directorio)

    for fichero in listaArchivos:
        if fichero[-3:] == kind1:
            print(fichero)
            texto1 = "convert -density %dx%d -scale %d%%  -quality %d%% " % (
                densidad, densidad, escala, quality)
            texto2 = texto1 + fichero + " " + fichero[:-3] + kind2
            print(texto2)
            os.system(texto2)


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
        color="YlGnBu",  # YlGnBu  RdBu
        interpolation='bilinear',  # 'bilinear', 'nearest'
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
    elif reduce_matrix is 'standard':
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

    id_fig = plt.figure(figsize=(5, 4.5))
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
                        titulos='',
                        title='',
                        figsize='',
                        logarithm=False,
                        normalize=False):
    """Draws several fields in subplots

    Parameters:
        fields (list): list with several scalar_fields_XY
        titulos (list): list with titles
        title (str): suptitle
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

    for i in sorted(range(num_dibujos)):
        c = fields[i]
        id_fig.add_subplot(col, fil, i + 1)
        extension = [c.x.min(), c.x.max(), c.y.min(), c.y.max()]

        image = np.abs(c.u)**2

        if logarithm is True:
            image = np.log(image + 1)

        if normalize == 'maximum':
            image = image / image.max()

        IDimage = plt.imshow(
            image,
            interpolation='bilinear',
            aspect='auto',
            origin='lower',
            extent=extension)
        plt.title(titulos[i], fontsize=24)
        plt.suptitle(title, fontsize=26)
        plt.axis('scaled')
        plt.axis(extension)
        plt.colorbar(orientation='horizontal', shrink=0.5)
        IDimage.set_cmap("gist_heat")


# def generate_video_proposal(u_s,
#                             kind,
#                             filename,
#                             logarithm,
#                             normalize,
#                             dimension='xz',
#                             wait=0.01 * seconds,
#                             step=1,
#                             fps=15,
#                             title='',
#                             artist='',
#                             comment=''):
#     """
#     draw profiles in a video fashion
#     Parameters:
#         kind = 'intensity', 'amplitude', 'phase'
#         kind_profile = 'transversal', 'longitudinal'
#         step: number of frames shown (if 1 shows all, if 2 1/2, ..)
#               for accelerating proposes in video
#         wait : (in seconds) time for slow down the video
#         logartihm, normalize: for normalization of the video
#         filename: ''         - shown in screen
#                   'name.avi' - performs a video
#     """
#     fig = plt.figure()
#
#     plt.ylim(I_drawing.min(), I_drawing.max())
#
#     writer = prepare_video(fps, title, artist, comment)
#
#     with writer.saving(fig, filename, 300):
#         if kind_profile == 'transversal':
#             for i in range(0, len(self.z), step):
#                 I_drawing = prepare_drawing(u_s[i].u, kind, logarithm,
#                                             normalize)
#
#                 h1.set_ydata(I_drawing[:, i])
#                 # plt.title("z={:6.2f}, i={}".format(round(self.z[i], 2), i))
#                 plt.draw()
#                 if filename is '':
#                     plt.pause(wait)
#                 else:
#                     print("{}/{}".format(i, len(self.z)))
#                     writer.grab_frame()
#         elif kind_profile == 'xy':
#             manager = plt.get_current_fig_manager()
#
#             for i in range(0, len(self.z), step):
#                 I_drawing = prepare_drawing(u_s[i].u, kind, logarithm,
#                                             normalize)
#                 im.set_array(I_drawing)
#                 manager.canvas.draw()
#                 # plt.title("z={:6.2f}, i={}".format(round(self.z[i], 2), i))
#                 plt.draw()
#                 if filename is '':
#                     plt.pause(wait)
#                 else:
#                     # print("{}/{}".format(i, len(self.z)))
#                     writer.grab_frame()


def change_image_size(image_name,
                      length='800x600',
                      nombre_final='prueba.png',
                      dpi=300):
    """cambia el tamaño con imageMagick
    convert image_name -resize '1000' -units 300 nombre_final.png
        - anchura 1000 - mantiene forma
    convert image_name -resize 'x200' nombre_final.png
        - height  200  - mantiene forma
    convert image_name -resize '100x200>' nombre_final.png
        - mantiene forma, lo que sea mayor
    convert image_name -resize '100x200<' nombre_final.png
        - mantiene forma, lo que sea menor
    convert image_name -resize '@1000000' nombre_final.png
        - mantiene la forma, con 1Mpixel
    convert image_name -resize '100x200!' nombre_final.png
        - obliga a tener el tamaño, no mantiene escala
    """
    texto = "convert {} -resize {} {}".format(image_name, length, nombre_final)
    print(texto)
    os.system(texto)


def extract_image_from_video(nombre_video=None,
                             num_frame="[0, ]",
                             nombre_final='prueba.png'):
    """saca images del video con imageMagick
    convert 'animacion.avi[15,]' animacion_frame.png
        - saca el frame 15 (solo el 15)
    convert 'animacion.avi[15]' animacion_frame.png
        - saca los 15 primeros frame los numera con _1
    convert 'animacion.avi[5,10]' animacion_frame.png
        - saca el frame 5 y el 10
    """

    texto = "convert '%s%s' %s" % (nombre_video, num_frame, nombre_final)
    print(texto)
    os.system(texto)


# def convert_video(path):
#     cmd = "'mencoder \"%s.avi\" -o \"%s.mp4\" -ovc lavc -oac mp3lame' %({},{})"
#     # %s es el nombre del archivo
#
#     folder = os.path.dirname(path)
#     files = os.listdir(path)
#
#     nvideos = 1
#
#     for video in files:
#         if video[-4:] == ".avi":
#             name = path + video[:-4]
#             orden_final = eval(cmd.format(name, name))
#             os.system(orden_final)
#             print("_____________________llevo %d videos" % nvideos)
#             nvideos = nvideos + 1
#         else:
#             print(video)


def normalize_draw(u, logarithm=False, normalize=False, cut_value=None):
    """
    Gets a filed and changes its caracteristics for drawing

    Parameters:
        u (field): field
        logarithm (bool): If True applies logarithm to image: np.log(u + 1)
        normalize (str or bool): False, 'mean', 'intensity'
    """
    if logarithm == 1:
        u = np.log(u + 1)
        # u = np.log(10 * u + 1)

    if normalize is False:
        pass
    elif normalize == 'maximum':
        u = u / (np.abs(u).max() + eps)
    elif normalize == 'mean':
        u = u / u.mean()

    else:
        print("no normalization: not False, 'maximum, 'mean'")

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
    elif reduce_matrix is 'standard':
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

    if verbose:
        print(("reduce_matrix_size: size ini = {}, size_final = {}".format(
            image_ini, image.shape)))
    return image

# !/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        utils_dxf.py
# Purpose:     Common functions to dxf files. It mainly uses ezdxf pachage
#
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Copyright:   (c) Luis Miguel Sanchez-Brea AOCG / UCM
# License:     GPLv3 License
# ----------------------------------------------------------------------
# EZDXF by Manfred Moitzi


from __future__ import annotations

import os
import sys

from PIL import Image

import ezdxf
from ezdxf import recover
from ezdxf import bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.layouts import Modelspace


from .utils_typing import npt, Any, NDArray,  NDArrayFloat, NDArrayComplex
from .import np, plt


def set_pixel_density(fig: plt.Figure, ax: plt.Axes, ppu: int):
    """_summary_

    Args:
        fig (plt.Figure): _description_
        ax (plt.Axes): _description_
        ppu (int): pixels per drawing unit.
    """

    xmin, xmax = ax.get_xlim()
    width = xmax - xmin
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    dpi = fig.dpi
    width_inch = width * ppu / dpi
    height_inch = height * ppu / dpi
    fig.set_size_inches(width_inch, height_inch)


def set_pixel_size(fig: plt.Figure, size: tuple[int, int]):
    """_summary_

    Args:
        fig (plt.Figure): _description_
        size (tuple[int, int]): _description_

    Returns:
        _type_: _description_
    """
    x, y = size
    fig.set_size_inches(x / fig.dpi, y / fig.dpi)
    return fig


def binarize(image, center_level=128):
    """_summary_

    Args:
        image (_type_): _description_
        center_level (int, optional): _description_. Defaults to 128.

    Returns:
        _type_: _description_
    """

    image_new = np.zeros_like(image)
    image_new[image < center_level] = 0
    image_new[image > center_level] = 1

    return image_new


def load_dxf(filename_dxf: str, num_pixels: tuple[int, int], verbose: bool = False):
    """_summary_

    Args:
        filename_dxf (_type_): _description_
        num_pixels (_type_): _description_
        filename_png (str, optional): _description_. Defaults to ''.


    Returns:
        _type_: _description_

    """
    # Example frame:
    #     frame: dict or bool = False,
    #     r0 = np.array((0*um, 0))
    #     extent_dxf = [-500*um, +500*um, -250*um, +250*um]

    # temporal, for debugging
    filename_png = ''
    has_draw = False

    try:
        doc, auditor = recover.readfile(filename_dxf)
    except IOError:
        print(f'Not a DXF file or a generic I/O error.')
        sys.exit(1)
    except ezdxf.DXFStructureError:
        print(f'{"Invalid or corrupted DXF file."}')
        sys.exit(2)

    if filename_png == '':
        filename_png2 = "temp.png"
    else:
        filename_png2 = filename_png

    msp = doc.modelspace()

    # The auditor.errors attribute stores severe errors,
    # which may raise exceptions when rendering.
    if not auditor.has_errors:
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        ctx.current_layout_properties.set_colors(bg='#000000')

        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=True)

        # set margins as you like (as relative factor of width and height)
        ax.margins(0)
        # export image with a size of 1000x600 pixels
        fig = set_pixel_size(fig, num_pixels)
        fig.savefig(filename_png2, facecolor='#000000', edgecolor='#FFFFFF')
        fig.clear()

    cache = bbox.Cache()
    # get overall bounding box
    bounding_box = bbox.extents(msp, cache=cache)

    p1 = bounding_box.extmin
    p2 = bounding_box.extmax

    p_min = np.array((p1[0], p1[1]))
    p_max = np.array((p2[0], p2[1]))

    im_frame = Image.open(filename_png2)

    np_frame = np.array(im_frame.getdata())

    im = np.asarray(np_frame[:, 0].reshape([num_pixels[1], num_pixels[0]]))

    image_new = binarize(im, 128)

    # if frame is not False:
    #     r0 = frame['r0']
    #     ext_dxf = frame['extension']
    #     points0 = np.array([(ext_dxf[0], ext_dxf[2]), (ext_dxf[1], ext_dxf[2]),
    #                         (ext_dxf[1], ext_dxf[3]), (ext_dxf[0], ext_dxf[3])])
    #     points0 = points0 + r0
    #     msp.add_lwpolyline(points0, close=True, dxfattribs={"color": 6})

    if verbose:
        print("p_min = ", p_min)
        print("p_max = ", p_max)
        print("frame size: ", im_frame.size)
        # print(im_frame.format)
        # print(im_frame.mode)
        # print(image_new.min(), image_new.max())

    if filename_png == '':
        os.remove(filename_png2)

    if has_draw:
        plt.figure()
        plt.imshow(image_new, cmap='gray')
        plt.colorbar()
        plt.clim(0, 1)

    return image_new, p_min, p_max, msp

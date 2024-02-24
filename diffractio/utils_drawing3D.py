# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for drawing with pyvista """


import numpy as np

import pyvista
import pyvista as pv


def draw(
    self,
    kind: str = "volume",
    variable: str = "refractive_index",
    has_grid: bool = False,
    filename="",
    **kwargs
):
    """_summary_

    Args:
        kind (str, optional): volume, clip, slices, projections. Defaults to 'volume'.
        variable (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
        has_grid (bool, optional): add grid. Defaults to False.
        filename (str, optional): saves images: html, png or svg. Defaults to ''.
    """

    # pv.set_jupyter_backend('server')
    # pv.set_jupyter_backend(None)

    print(kwargs)
    opacity = kwargs["opacity"]
    dimensions = kwargs["dimensions"]
    scale = kwargs["scale"]
    cmap = kwargs["cmap"]
    spacing = kwargs["spacing"]
    pos_centers = kwargs["pos_centers"]
    pos_slices = kwargs["pos_slices"]
    cpos = kwargs["cpos"]

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing)

    intensity = self.intensity()
    intensity /= intensity.max()

    n = self.n

    if variable == "intensity":
        data = intensity
    elif variable == "refractive_index":
        data = n

    if kind == "volume":
        grid["scalars"] = data.flatten()
        pl = pyvista.Plotter()
        
        if cpos != None:
            pl.camera_position = cpos
            reset_camera = False

        pl.set_scale(
            xscale=scale[0],
            yscale=scale[1],
            zscale=scale[2],
            reset_camera=reset_camera,
            render=True,
        )
        vol = pl.add_volume(data, cmap=cmap, opacity=opacity, shade=False)
        vol.prop.interpolation_type = "linear"

    elif kind == "clip":
        pl = pyvista.Plotter()
        grid["scalars"] = data.flatten()

        pl.add_volume_clip_plane(grid, normal="x", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="y", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="-z", opacity=opacity, cmap=cmap)
        pl.set_scale(
            xscale=scale[0],
            yscale=scale[1],
            zscale=scale[2],
            reset_camera=True,
            render=True,
        )

    elif kind == "slices":
        pl = pyvista.Plotter()
        grid["scalars"] = data.flatten()

        slice = grid.slice_orthogonal()
        dargs = dict(cmap=cmap)
        pl.add_mesh(slice, **dargs)
        pl.set_scale(
            xscale=scale[0],
            yscale=scale[1],
            zscale=scale[2],
            reset_camera=True,
            render=True,
        )

    elif kind == "projections":
        pl = pv.Plotter(shape=(2, 2))
        dargs = dict(cmap=cmap)
        grid["scalars"] = data.flatten()

        slice1 = grid.slice_orthogonal(x=0, z=0)
        slice2 = grid.slice_orthogonal(x=0, y=0)
        slice3 = grid.slice_orthogonal(y=0, z=0)
        slice4 = grid.slice_orthogonal(x=pos_centers[2], z=pos_centers[2])

        # XYZ - show 3D scene first
        pl.subplot(1, 1)
        pl.add_mesh(slice4, **dargs)
        # XY
        pl.subplot(0, 0)
        pl.add_mesh(slice1, **dargs)
        # p.show_grid()
        pl.camera_position = "xz"
        pl.enable_parallel_projection()
        # ZY
        pl.subplot(0, 1)
        pl.add_mesh(slice2, **dargs)
        # p.show_grid()
        pl.camera_position = "xy"
        pl.enable_parallel_projection()
        # XZ
        pl.subplot(1, 0)
        pl.add_mesh(slice3, **dargs)
        # p.show_grid()
        pl.camera_position = "yz"
        pl.enable_parallel_projection()

    # pl.view_isometric()
    # pl.show_axes()
    # pl.show_bounds()

    if has_grid is True:
        pl.show_grid()

    pl.show()

    if filename != "":
        if filename[-3:] == "svg":
            pl.save_graphic(filename)
        elif filename[-4:] == "html":
            pl.export_html(filename)
        elif filename[-3:] == "png":
            pl.screenshot(filename)


def video_isovalue(self, filename: str, variable: str = "refractive_index", **kwargs):
    """_summary_

    Args:
        filename (str): _description_. Defaults to ''.
        variable (str, optional): "intensity" or "refractive_index". Defaults to 'refractive_index'.
    """
    # pv.set_jupyter_backend('server')
    # pv.set_jupyter_backend(None)

    print(kwargs)
    opacity = kwargs["opacity"]
    dimensions = kwargs["dimensions"]
    scale = kwargs["scale"]
    cmap = kwargs["cmap"]
    spacing = kwargs["spacing"]
    pos_centers = kwargs["pos_centers"]
    pos_slices = kwargs["pos_slices"]

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing)

    if variable == "intensity":
        intensity = self.intensity()
        intensity /= intensity.max()

        data = intensity
    elif variable == "refractive_index":
        data = self.n

    grid["scalars"] = data.flatten()

    pl = pv.Plotter()
    pl.set_scale(
        xscale=scale[0],
        yscale=scale[1],
        zscale=scale[2],
        reset_camera=True,
        render=True,
    )

    vol = pl.add_volume(grid, opacity=opacity, cmap=cmap)
    vol.prop.interpolation_type = "linear"

    values = np.linspace(0.1 * data.max(), data.max(), num=25)
    surface = grid.contour(values[:1])

    surfaces = [grid.contour([v]) for v in values]

    surface = surfaces[0].copy()

    plotter = pv.Plotter(off_screen=True)
    # Open a movie file
    plotter.open_gif(filename)

    # Add initial mesh
    plotter.add_mesh(
        surface,
        opacity=0.5,
        cmap=cmap,
        clim=grid.get_data_range(),
        show_scalar_bar=False,
    )
    # Add outline for reference
    plotter.add_mesh(grid.outline_corners(), color="r")

    # print('Orient the view, then press "q" to close window and produce movie')

    # initial render and do NOT close
    plotter.show(auto_close=True)

    # Run through each frame
    for surf in surfaces:
        surface.copy_from(surf)
        plotter.write_frame()  # Write this frame
    # Run through backwards
    for surf in surfaces[::-1]:
        surface.copy_from(surf)
        plotter.write_frame()  # Write this frame

    # Be sure to close the plotter when finished
    plotter.close()
    pl.close()


def show_stl(filename):

    mesh = pv.read(filename)
    mesh.plot()

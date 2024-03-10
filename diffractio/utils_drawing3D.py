# !/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for drawing with pyvista """


import numpy as np

import pyvista
import pyvista as pv
import collections
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.helpers import wrap

def load_stl(filename, has_draw=True, verbose=True):

    mesh = pv.read(filename)
    
    if has_draw:
        mesh.plot()
        
    if verbose:
        print(filename)
        print("bounds")
        print(mesh.bounds)
        print("volume")
        print(mesh.volume)
        print("center")
        print(mesh.center)
    return mesh, filename, mesh.bounds



def voxelize_volume_diffractio(self, mesh, refractive_index,  check_surface=True):
    """Voxelize mesh to create a RectilinearGrid voxel volume.

    Creates a voxel volume that encloses the input mesh and discretizes the cells
    within the volume that intersect or are contained within the input mesh.
    ``InsideMesh``, an array in ``cell_data``, is ``1`` for cells inside and ``0`` outside.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Mesh to voxelize.


    check_surface : bool, default: True
        Specify whether to check the surface for closure. If on, then the
        algorithm first checks to see if the surface is closed and
        manifold. If the surface is not closed and manifold, a runtime
        error is raised.

    Returns
    -------
    pyvista.RectilinearGrid
        RectilinearGrid as voxelized volume with discretized cells.

    See Also
    --------
    pyvista.voxelize
    pyvista.DataSetFilters.select_enclosed_points

    """
    mesh = wrap(mesh)
    
    # check and pre-process input mesh
    surface = mesh.extract_geometry()  # filter preserves topology
    if not surface.faces.size:
        # we have a point cloud or an empty mesh
        raise ValueError('Input mesh must have faces for voxelization.')
    if not surface.is_all_triangles:
        # reduce chance for artifacts, see gh-1743
        surface.triangulate(inplace=True)

 
    # Create a RectilinearGrid
    voi = pyvista.RectilinearGrid(self.x, self.y, self.z)

    # get part of the mesh within the mesh's bounding surface.
    selection = voi.select_enclosed_points(surface, tolerance=0.0, check_surface=check_surface)
    mask_vol = selection.point_data['SelectedPoints'].view(np.bool_)

    # Get voxels that fall within input mesh boundaries
    cell_ids = np.unique(voi.extract_points(np.argwhere(mask_vol))["vtkOriginalCellIds"])
    # Create new element of grid where all cells _within_ mesh boundary are
    # given new name 'MeshCells' and a discrete value of 1
    voi['InsideMesh'] = np.zeros(voi.n_cells)
    voi['InsideMesh'][cell_ids] = refractive_index
    
    print(type(mask_vol))
    # mask_vol = mask_vol.reshape(self.x, self.y, self.z)
    volume_n = self.n_background + (refractive_index-self.n_background)*mask_vol
    self.n = volume_n.reshape(len(self.y), len(self.x), len(self.z))
    
    print(self.n.shape)

    return self, voi #, selection, mask_vol, cell_ids, volume_n

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
    spacing = kwargs["spacing"]
    pos_centers = kwargs["pos_centers"]
    pos_slices = kwargs["pos_slices"]
    cpos = kwargs["cpos"]

    grid = pv.ImageData(dimensions=dimensions, spacing=spacing)

    intensity = self.intensity()
    intensity = intensity/intensity.max()

    n = self.n

    if variable == "intensity":
        data = intensity
        cmap = kwargs["cmap_intensity"]

    elif variable == "refractive_index":
        data = n
        cmap = kwargs["cmap_n"]
        
    data = data.reshape((len(self.y),len(self.x),len(self.z)))
    grid["scalars"] =  np.transpose(data,axes=(2,1,0)).flatten()

    if kind == "volume":
        pl = pyvista.Plotter()
        
        if cpos != None:
            pl.camera_position = cpos
            reset_camera = False

        pl.set_scale(
            xscale=scale[1],
            yscale=scale[0],
            zscale=scale[2],
            reset_camera=reset_camera,
            render=True,
        )
        vol = pl.add_volume(data, cmap=cmap, opacity=opacity, shade=False)
        vol.prop.interpolation_type = "linear"

    elif kind == "clip":
        pl = pyvista.Plotter()
        pl.add_volume_clip_plane(grid, normal="x", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="y", opacity=opacity, cmap=cmap)
        pl.add_volume_clip_plane(grid, normal="-z", opacity=opacity, cmap=cmap)
        pl.set_scale(
            xscale=scale[1],
            yscale=scale[0],
            zscale=scale[2],
            reset_camera=True,
            render=True)

    elif kind == "slices":
        pl = pyvista.Plotter()
        slice = grid.slice_orthogonal()
        dargs = dict(cmap=cmap)
        pl.add_mesh(slice, **dargs)
        pl.set_scale(
            xscale=scale[1],
            yscale=scale[0],
            zscale=scale[2],
            reset_camera=True,
            render=True,
        )

    elif kind == "projections":
        pl = pv.Plotter(shape=(2, 2))
        dargs = dict(cmap=cmap)
        slice1 = grid.slice_orthogonal(x=0, z=0)
        slice2 = grid.slice_orthogonal(x=0, y=0)
        slice3 = grid.slice_orthogonal(y=0, z=0)
        slice4 = grid.slice_orthogonal()

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
        xscale=scale[1],
        yscale=scale[0],
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

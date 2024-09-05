# !/usr/bin/env python3

"""
This module generates Scalar_mask_XYZ class for definingn masks. Its parent is scalar_fields_XYZ.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.z - z positions of the field
    * self.u - field XYZ
    * self.n - refractive index XYZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * object_by_surfaces
    * sphere
    * square
    * cylinder
"""

# flake8: noqa


from .utils_typing import npt, Any, NDArray, floating, NDArrayFloat, NDArrayComplex


from . import degrees, np, um
from .scalar_fields_XYZ import Scalar_field_XYZ
from .utils_drawing3D import load_stl, voxelize_volume_diffractio

class Scalar_mask_XYZ(Scalar_field_XYZ):

    def __init__(self, x: NDArrayFloat | None = None, y: NDArrayFloat | None = None,
                 z: NDArrayFloat | None = None, wavelength: float | None = None,
                 n_background: float = 1., info: str = ""):
        super().__init__(x, y, z, wavelength, n_background, info)
        self.type = 'Scalar_mask_XYZ'

    def object_by_surfaces(self,
                           r0: list[floating],
                           refractive_index: floating,
                           Fs,
                           angles,
                           v_globals={}):
        """ TODO
        Mask defined by n surfaces given in array Fs={f1, f2,    h(x,y,z)=f1(x,y,z)*f2(x,y,z)*....*fn(x,y,z)


        Args:
            rotation_point (float, float, float): location of the mask
            refractive_index (float, str): can be a number or a function n(x, y,z)
            Fs (list): condtions as str that will be computed using eval
            array1 (numpy.array): array (x,y,z) that delimits the second surface
            angle (float): angle of rotation (radians)
            v_globals (dict): dict with global variables
            verbose (bool): shows data if true

        """

        if angles not in ('', None, []):
            psi, phi, sigma = angles
            Xrot, Yrot, Zrot = self.__rotate__(psi, phi, sigma, r0)
        else:
            Xrot = self.X
            Yrot = self.Y
            Zrot = self.Z

        v_locals = {'self': self, 'np': np, 'degrees': degrees, 'um': um}
        v_locals['Xrot'] = Xrot
        v_locals['Yrot'] = Yrot
        v_locals['Zrot'] = Zrot

        conditions = []
        for fi in Fs:
            result_condition = eval(fi, v_globals, v_locals)
            conditions.append(result_condition)

        # Transmitancia de los points interiores
        ipasa = conditions[0]
        for cond in conditions:
            ipasa = ipasa & cond

        self.n[ipasa] = refractive_index
        return ipasa

    def sphere(self, r0: list[floating], radius: list[floating], refractive_index: floating, angles):
        """ Insert a sphere in background. If it is something else previous, it is removed.

            Args:
                r0:(x0, y0, z0) Location of sphere, for example (0 * um, 0*um, 0 * um)
                radius: (rx, ry, rz) Radius of sphere. It can be a ellipsoid. If radius is a number, then it is a sphere
                refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
        """
        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius, radius)

        x0, y0, z0 = r0
        radiusx, radiusy, radiusz = radius

        ipasa = (self.X - x0)**2 / radiusx**2 + (
            self.Y - y0)**2 / radiusy**2 + (self.Z - z0)**2 / radiusz**2 < 1
        self.n[ipasa] = refractive_index

        return ipasa

    def square(self,
               r0: list[floating],
               length: list[floating],
               refractive_index: floating,
               angles=None,
               rotation_point: list[floating] = None):
        """ Insert a rectangle in background. If something previous, is removed.

        Args:
            r0 (float, float, float): (x0, y0,z0) Location of the rectangle, for example (0*um, 0*um, 0*um)
            size (float, float, float): x,y,z size of the rectangle
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            angle (float): angle of rotation of the semi-plane, in radians
            rotation_point (float, float, float). Rotation point
        """

        if rotation_point is None:
            rotation_point = r0

        if isinstance(r0, (float, int, complex)):
            r0 = (r0[0], r0[0], r0[0])
        if len(length) == 1:
            length = (length[0], length[0], length[0])

        x0, y0, z0 = r0
        lengthx, lengthy, lengthz = length

        ipasax1 = self.X >= x0 - lengthx / 2
        ipasax2 = self.X <= x0 + lengthx / 2
        ipasay1 = self.Y >= y0 - lengthy / 2
        ipasay2 = self.Y <= y0 + lengthy / 2
        ipasaz1 = self.Z >= z0 - lengthz / 2
        ipasaz2 = self.Z <= z0 + lengthz / 2
        ipasa = ipasax1 * ipasax2 * ipasay1 * ipasay2 * ipasaz1 * ipasaz2
        self.n[ipasa] = refractive_index

        return ipasa

    def cylinder(self, r0: list[floating], radius: list[floating], length: float,
                 refractive_index: float, axis: list[floating], angle: floating):
        """ Insert a cylinder in background. If something previous, is removed.

        Args:
            r0 (float, float, float): (x0, y0,z0) Location of the rectangle, for example (0*um, 0*um, 0*um)
            radius (float,float): x,y, size of the circular part of cylinder
            length (float): length of cylidner
            refractive_index (float, str): refractive index , for example: 1.5 + 1.0j
            axis (float float, float): axis direction
            angle (float): angle of rotation of the semi-plane, in radians
        """
        # si solamente hay un numero, es que las posiciones y radius
        # son los mismos para ambos
        if isinstance(r0, (float, int, complex)):
            r0 = (r0[0], r0[0], r0[0])
        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius)

        x0, y0, z0 = r0
        radiusx, radiusy = radius

        ipasar = (self.X - x0)**2 / radiusx**2 + (self.Y -
                                                  y0)**2 / radiusy**2 <= 1
        ipasaz1 = self.Z >= z0 - length / 2
        ipasaz2 = self.Z <= z0 + length / 2
        ipasa = ipasar * ipasaz1 * ipasaz2
        """
        FIXME: not working

        # psi,phi,sigma=angles
        # if not (psi ==0 and phi==0 and sigma==0):
        if angle != 0:
            # Xrot, Yrot, Zrot = self.__rotate__(psi, phi, sigma)
            Xrot, Yrot, Zrot = self..__rotate_axis__(axis, angle)
        else:

            Xrot=self.X
            Yrot=self.Y
            Zrot=self.Z
          """

        self.n[ipasa] = refractive_index

        return ipasa




    def stl(self, filename: str, refractive_index: float, Dx: float | None = None, Dy: float | None = None, 
            Dz: float | None = None, has_draw: bool = False, verbose: bool = False):
        """
        stl _summary_

        _extended_summary_

        Args:
            filename (str): _description_
            refractive_index (float): _description_
            Dx (float | None, optional): _description_. Defaults to None.
            Dy (float | None, optional): _description_. Defaults to None.
            Dz (float | None, optional): _description_. Defaults to None.
            has_draw (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        mesh = load_stl(filename, has_draw=has_draw)
        
        bounds = mesh.bounds

        if Dx is None:
            x = self.x
        else:
            x = np.linspace(bounds[0]-Dx[0], bounds[1]+Dx[1], len(self.x))

        if Dy is None:
            y = self.y
        else:
            y = np.linspace(bounds[2]-Dy[0], bounds[3]+Dy[1], len(self.y))

        if Dz is None:
            z = self.z
        else:
            z = np.linspace(bounds[4]-Dz[0], bounds[5]+Dz[1], len(self.z))

        self.x = x
        self.y = y
        self.z = z
        
        self.X, self.Y, self.Z = np.meshgrid(x, y, z)

        # x_center = (x[-1]+x[0])/2
        # y_center = (y[-1]+y[0])/2
        # z_center = (z[-1]+z[0])/2

        # len_x = len(x)
        # len_y = len(y)
        # len_z = len(z)

        self, voi= voxelize_volume_diffractio(self, mesh, refractive_index = refractive_index)

        # params_pyvista = {
        # 'opacity': [0,0.2,0.5, 1], #'sigmoid',
        # 'dimensions': (len(y),len(x), len(z)) ,
        # 'scale': (1, 1, 1),
        # 'cmap': 'hot',
        # 'spacing' : np.array((y[1]-y[0],x[1]-x[0],z[1]-z[0])),
        # 'pos_slices': np.array([(x_center, z_center),(x_center, y_center), (y_center, z_center)]), #xz, xy, yz, z0
        # 'pos_centers': np.array([y_center, x_center, z_center]),
        
        # 'cpos': [(540, -617, 180),
        #         (128, 126., 111.),
        #         (-0, 0 ,0)],
        
        # }

        return voi, mesh, bounds
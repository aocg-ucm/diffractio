# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module generates Scalar_mask_XYZ class for definingn masks. Its parent is scalar_fields_XYZ.

The main atributes are:
    * self.x - x positions of the field
    * self.y - y positions of the field
    * self.z - z positions of the field
    * self.u - field XYZ
    * self.n - refraction index XYZ
    * self.wavelength - wavelength of the incident field. The field is monochromatic

The magnitude is related to microns: `micron = 1.`

*Class for unidimensional scalar masks*

*Functions*
    * object_by_surfaces
    * sphere
    * square
    * cylinder
"""

from . import degrees, np, um
from .scalar_fields_XYZ import Scalar_field_XYZ


class Scalar_mask_XYZ(Scalar_field_XYZ):
    def __init__(self, x, y, z, wavelength, n_background=1., info=''):
        # print("init de Scalar_mask_XYZ")
        super(self.__class__, self).__init__(x, y, z, wavelength, n_background,
                                             info)

    def object_by_surfaces(self,
                           r0,
                           refraction_index,
                           Fs,
                           angles,
                           v_globals={}):
        """Mask defined by n surfaces given in array Fs={f1, f2,    h(x,y,z)=f1(x,y,z)*f2(x,y,z)*....*fn(x,y,z)


        Parameters:
            rotation_point (float, float, float): location of the mask
            refraction_index (float, str): can be a number or a function n(x, y,z)
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

        self.n[ipasa] = refraction_index
        return ipasa

    def sphere(self, r0, radius, refraction_index, angles):
        """ Insert a sphere in background. If it is something else previous, it is removed.

            Parameters:
                r0:(x0, y0, z0) Location of sphere, for example (0 * um, 0*um, 0 * um)
                radius: (rx, ry, rz) Radius of sphere. It can be a ellipsoid. If radius is a number, then it is a sphere
                        refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
        """
        if isinstance(radius, (float, int, complex)):
            radius = (radius, radius, radius)

        x0, y0, z0 = r0
        radiusx, radiusy, radiusz = radius

        ipasa = (self.X - x0)**2 / radiusx**2 + (
            self.Y - y0)**2 / radiusy**2 + (self.Z - z0)**2 / radiusz**2 < 1
        self.n[ipasa] = refraction_index

        return ipasa

    def square(self,
               r0,
               length,
               refraction_index,
               angles=None,
               rotation_point=None):
        """ Insert a rectangle in background. If something previous, is removed.

        Parameters:
            r0 (float, float, float): (x0, y0,z0) Location of the rectangle, for example (0*um, 0*um, 0*um)
            size (float, float, float): x,y,z size of the rectangle
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
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
        self.n[ipasa] = refraction_index

        return ipasa

    def cylinder(self,
                 r0,
                 radius,
                 length,
                 refraction_index,
                 axis,
                 angle):
        """ Insert a cylinder in background. If something previous, is removed.

        Parameters:
            r0 (float, float, float): (x0, y0,z0) Location of the rectangle, for example (0*um, 0*um, 0*um)
            radius (float,float): x,y, size of the circular part of cylinder
            length (float): length of cylidner
            refraction_index (float, str): refraction index , for example: 1.5 + 1.0j
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
        TODO: no funciona

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

        self.n[ipasa] = refraction_index

        return ipasa

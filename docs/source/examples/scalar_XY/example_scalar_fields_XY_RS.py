# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clases scalar_fields_XY y Fuentes2D con las fuentes utilizadas para propagacion
"""

from diffractio import degrees, mm, np, plt, sp, um
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY


def example_RS_slit():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
    # u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.slit(x0=0, size=10 * um, angle=0 * degrees)

    u2 = u1 * t1
    u2.draw()
    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=100 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_double_slit():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
    # u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.double_slit(x0=0, size=2 * um, separation=10 * um, angle=0 * degrees)

    u2 = u1 * t1
    u2.draw()
    u3 = u2.RS(z=20 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=200 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_square():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
    # u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.square(r0=(0 * um, 0 * um), size=(25 * um, 25 * um), angle=0 * degrees)

    u2 = u1 * t1
    u2.draw()
    u3 = u2.RS(z=100 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=500 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_circle():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.circle(
        r0=(0 * um, 0 * um), radius=(25 * um, 25 * um), angle=0 * degrees)

    u2 = u1 * t1
    u2.draw()
    u3 = u2.RS(z=100 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=500 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_ring():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.ring(
        r0=(0 * um, 0 * um),
        radius1=(5 * um, 10 * um),
        radius2=(10 * um, 20 * um),
        angle=45)

    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=100 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=500 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_cross():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.cross(r0=(0 * um, 0 * um), size=(75 * um, 25 * um), angle=45 * degrees)

    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_two_levels():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.two_levels(level1=0, level2=.5, xcorte=0, angle=0)

    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_image():
    num_pixels = 512

    length = 100 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)

    t1.image(filename="../../diffractio/images/lenaColor.png", invert=False)
    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=50 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_gray_scale():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.gray_scale(num_levels=128, levelMin=0, levelMax=1)

    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_mask_phase_1():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.mask_from_function(
        r0=(0 * um, 0 * um),
        index=1.5,
        f1='4*degrees*self.Y',
        f2='4*degrees*self.X',
        v_globals=None,
        radius=(50 * um, 50 * um),
        mask=True)

    u2 = u1 * t1
    u2.draw(kind='field')

    u3 = u2.RS(z=25 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_mask_phase_2():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    f1 = "1*degrees*self.X"
    f1 = "np.zeros_like(self.X,dtype=float)"
    f1 = "R1-h1+np.sqrt(R1**4-(self.X-x0)**4-(self.Y-y0)**4)"
    f2 = "R2-h2+np.sqrt(R2**2-(self.X)**2-(self.Y)**2)"
    v_globals = {
        'R1': 5 * mm,
        'R2': 1 * mm,
        'x0': 0 * um,
        'y0': 25 * um,
        'h1': 4 * mm,
        'h2': -1 * mm
    }
    index = 1.5
    print(v_globals)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.mask_from_function(
        r0=(0 * um, 0 * um),
        index=index,
        f1=f1,
        f2=f2,
        v_globals=v_globals,
        radius=(100 * um, 100 * um),
        mask=True)

    u2 = u1 * t1
    u2.draw(kind='field')

    u3 = u2.RS(z=1000 * um, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=3000 * um, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_prism():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.prism(
        r0=(0, 0),
        index=1.5,
        angle_wedge_x=1 * degrees,
        angle_wedge_y=0 * degrees,
        angle=0 * degrees)

    u2 = u1 * t1
    u2.draw(kind='phase')

    u3 = u2.RS(z=1 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=2 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_lens():
    num_pixels = 512

    length = 200 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.lens(
        r0=(0 * um, 0 * um),
        radius=(100 * um, 100 * um),
        focal=(5 * mm, 10 * mm),
        angle=0 * degrees)

    u2 = u1 * t1
    u2.draw()

    u3 = u2.RS(z=5 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=7 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=10 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_lens_fresnel():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.fresnel_lens(
        r0=(0 * um, 0 * um),
        radius=(125 * um, 125 * um),
        focal=(1 * mm, 1 * mm),
        angle=45 * degrees,
        kind='amplitude',
        phase=np.pi,
        mask=True)
    u2 = u1 * t1
    u2.draw(kind='amplitude')
    u3 = u2.RS(z=1 * mm, new_field=True)
    u3.draw(kind='field')

    t1.fresnel_lens(
        r0=(0 * um, 0 * um),
        radius=(125 * um, 125 * um),
        focal=(1 * mm, 1 * mm),
        angle=0 * degrees,
        kind='phase',
        phase=np.pi,
        mask=True)
    u2 = u1 * t1
    u2.draw(kind='phase')
    u3 = u2.RS(z=1 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_lens_billet():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)

    t1.lens_billet(
        r0=(0 * um, 0 * um),
        radius=(125 * um, 125 * um),
        focal=(2 * mm, 2 * mm),
        angle=0 * degrees,
        radius_agujero=5 * um)
    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=2 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=4 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_biprism_fresnel():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)

    t1.biprism_fresnel(
        r0=(0 * um, 0 * um), ancho=125 * um, height=5 * um, n=1.5)
    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=1.25 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=2.5 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_axicon():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.axicon(r0=(0 * um, 0 * um), radius=125 * um, height=2 * um, n=1.5)

    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=1.25 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=2.5 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_laguerre_gauss_spiral():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.laguerre_gauss_spiral(
        kind='intensity', l=4, r0=(0 * um, 0 * um), w0=20 * um, z=0.01 * um)

    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=5 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=10 * mm, new_field=True)
    u3.draw(kind='field')

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.laguerre_gauss_spiral(
        kind='phase', l=4, r0=(0 * um, 0 * um), w0=20 * um, z=0.01 * um)

    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=2.5 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=10 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_forked_grating():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.forked_grating(
        r0=(0 * um, 0 * um), period=10 * um, l=3, alpha=2, angle=0 * degrees)

    zt = 2 * 10**2 / wavelength

    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=zt, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=2 * zt, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_roughness():
    num_pixels = 512

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.roughness(t=(25 * um, 50 * um), s=2 * um)

    u2 = u1 * t1
    u2.draw(kind='field')
    u3 = u2.RS(z=5 * mm, new_field=True)
    u3.draw(kind='field')

    u3 = u2.RS(z=10 * mm, new_field=True)
    u3.draw(kind='field')
    return True


def example_RS_redBlazed():
    num_pixels = 1024

    length = 250 * um
    x0 = np.linspace(-length / 2, length / 2, num_pixels)
    y0 = np.linspace(-length / 2, length / 2, num_pixels)
    wavelength = 0.6238 * um

    u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
    u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

    t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t1.blazed_grating(
        period=10 * um,
        height=1 * um,
        index=1.5,
        x0=0 * um,
        angle=45 * degrees)

    t2 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t2.circle(r0=(0, 0), radius=(50 * um, 50 * um), angle=0 * degrees)

    t3 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
    t3.lens(
        r0=(0, 0),
        radius=(125 * um, 125 * um),
        focal=(1 * mm, 1 * mm),
        angle=0,
        mask=False)

    u2 = u1 * t1 * t2 * t3
    u2.draw(kind='field')
    u3 = u2.RS(z=200 * um, new_field=True)
    u3.draw(kind='field', logarithm=True)

    u3 = u2.RS(z=600 * um, new_field=True)
    u3.draw(kind='field', logarithm=True)
    return True


example_RS_roughness()

example_RS_redBlazed()
plt.show()

================================================
Python Diffraction-Interference module
================================================


.. image:: https://img.shields.io/pypi/v/diffractio.svg
        :target: https://pypi.org/project/diffractio/

.. image:: https://img.shields.io/travis/optbrea/diffractio.svg
        :target: https://bitbucket.org/optbrea/diffractio/src/master/

.. image:: https://readthedocs.org/projects/diffractio/badge/?version=latest
        :target: https://diffractio.readthedocs.io/en/latest/
        :alt: Documentation Status


* Free software: MIT license

* Documentation: https://diffractio.readthedocs.io/en/latest/


.. image:: logo.png
   :width: 75
   :align: right


Features
----------------------

Diffratio is a Python library for Diffraction and Interference Optics.

It implements Scalar and paraxial vector Optics. The main algorithms used are Rayleigh Sommerfeld (RS), Beam Propagation Method (BPM) and Fast Fourier Transform (FFT). When possible, multiprocessing is implemented for a faster computation.

The scalar propagations techniques are implemented to:

* X - fields are defined in the x axis.
* XZ - fields are defined in the xz plane, being z the propagation direction.
* XY - fields are defined in the xy transversal plane.
* XYZ - fields are defined in the xyz volume.
* vector_paraxial_XY - Ex and Ey electric field components are defined, which allows polarization analysis.

Each technique present three modules:

* sources: Generation of light.
* masks: Masks and Diffractive Optical elements.
* fields:  Propagation techniques, parameters and general functions.

The paraxial vector propagation techniques are implemented to:

* XY - fields are defined in the xy transversal plane.


Sources
========

One main part of this software is the generation of optical fields such as:

* Plane waves.
* Spherical waves.
* Gaussian beams.
* Bessel beams.
* Aberrated beams.

Also, in the XY module the following sources are defined:

* Vortex beams.
* Laguerre beams.
* Hermite-Gauss beams.
* Zernike beams.
* Bessel beams.

.. image:: source.png
   :width: 400


Masks
=============

Another important part of Diffractio is the generation of masks and Diffractive Optical Elements such as:

* Slits, double slits
* Lenses, diffractive lenses, aspherical lenses.
* Gratings, prisms, biprism
* Rough surfaces, dust ks are defined as plane. However, in the XZ and XYZ frames, volumetric mask are also defined.


.. image:: mask1.png
   :height: 300

.. image:: mask2.png
   :height: 350


Fields
=========

In these module, algorithms for propagation of light are implemented. We have implemented the following algorithms for light propagation:

* **Rayleigh-Sommerfeld (RS)** which allows in a single step to propagate to a near or far observation plane, which allows fast computations. The fields and the masks must be defined in a plane.

* **Beam propagation method (BPM)** which allows to analyze the propation of light in volumetric elements, such as spheres, cylinders and other complex forms.

* **Fast Fourier Transform (FFT)** which allows, in a single step to determine the field at the far field.

* **Plane Wave Descomposition (PWD)**.

* **Wave Propagation Method (PWD)**.

* **Vector Rayleigh-Sommerfeld (VRS)**.

* **Vector Wave Propagation Method (VPWD)**.


The fields, masks and sources can be stored in files.

Also drawings can be easily obtained, for intensity, phase, fields, etc.

In some modules, videos can be generated for a better analysis of optical fields.

.. image:: propagation.png
   :width: 400




Paraxial vector beams
==================================

Here, we implement new classes where the fields E_x and E_y are generated and propagted using Rayleigh-Sommerfeld approach.
Also, simple and complex polarizing masks can be created.

**Ex and Ey fields**

.. image:: vector_gauss_radial_fields.png
   :width: 700

**Polarization: Stokes parameters**

.. image:: vector_gauss_radial_stokes.png
  :width: 700


Other features
=================

* Intensity, MTF and other parameters are obtained from the optical fields.

* Fields can be added and interference is produced. Masks can be multiplied, added and substracted in order to make complex structures

* Resampling fields in order to analyze only areas of interest.

* Save and load data for future analysis.

* Rayleigh-Sommerfeld implementation is performed in multiprocessing for fast computation.

* Polychromatic and extended source problems can also be analyzed using multiprocessing.



Authors
---------------------------

* Luis Miguel Sanchez Brea <optbrea@ucm.es>


    **Universidad Complutense de Madrid**,
    Faculty of Physical Sciences,
    Department of Optics
    Plaza de las ciencias 1,
    ES-28040 Madrid (Spain)

.. image:: logoUCM.png
   :width: 125
   :align: right



Citing
---------------------------

L.M. Sanchez Brea, "Diffratio, python module for diffraction and interference optics", https://pypi.org/project/diffractio/ (2019)


References
---------------------------

**Propagation algorithms**:

* J. W. Goodman, Introduction to Fourier optics. McGraw-Hill, 1996.
* Shen, F. & Wang, A. Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula. Appl. Opt. 45, 1102–1110 (2006).
* Ye, H. et al. Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: Vectorial Rayleigh-Sommerfeld method. Laser Phys. Lett. 10, (2013).
* Fertig, M. & Brenner, K.-H. Vector wave propagation method. J. Opt. Soc. Am. A 27, 709 (2010).
* Schmidt, S. et al. Wave-optical modeling beyond the thin-element-approximation. Opt. Express 24, 30188 (2016).
* Schmidt, S., Thiele, S., Herkommer, A., Tünnermann, A. & Gross, H. Rotationally symmetric formulation of the wave propagation method-application to the straylight analysis of diffractive lenses. Opt. Lett. 42, 1612 (2017).
* Z. Qiwen, Vectorial optical fields: Fundamentals and applications. World scientific, 2013.
* B. E. Saleh y M. C. Teich, Fundamentals of photonics. John Wiley & Sons, 2019.
* J. A. Ogilvy, Theory of Wave Scattering from Random Rough Surfaces.Adam Hilger, 1991.
* "Numerical Methods in Photonics Lecture Notes".  http://ecee.colorado.edu/~mcleod/teaching/nmip/lecturenotes.html.
* Beam width: https://en.wikipedia.org/wiki/Beam_diameter




Credits
---------------------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

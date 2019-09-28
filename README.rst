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

It implements Scalar Optics. The main algorithms used are Rayleigh Sommerfeld (RS), Beam Propagation Method (BPM) and Fast Fourier Transform (FFT). When possible, multiprocessing is implemented for a faster computation.

The scalar propagations techniques are implemented to:

* X - fields are defined in the x axis.
* XZ - fields are defined in the xz plane, being z the propagation direction.
* XY - fields are defined in the xy transversal plane.
* XYZ - fields are defined in the xyz volume.



Each technique present three modules:

* sources: Generation of light.
* masks: Masks and Diffractive Optical elements.
* fields:  Propagation techniques, parameters and general functions.


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
   :width: 400

.. image:: mask2.png
   :width: 400


Fields
=========

In these module, algorithms for propagation of light are implemented. We have implemented the following algorithms for light propagation:

* **Rayleigh-Sommerfeld (RS)** which allows in a single step to propagate to a near or far observation plane, which allows fast computations. The fields and the masks must be defined in a plane.

* **Beam propagation method (BPM)** which allows to analyze the propation of light in volumetric elements, such as spheres, cylinders and other complex forms.

* **Fast Fourier Transform (FFT)** which allows, in a single step to determine the field at the far field.



The fields, masks and sources can be stored in files,

Also drawings can be easily obtained, for intensity, phase, fields, etc.

In some modules, videos can be generated for a better analysis of optical fields.

.. image:: propagation.png
   :width: 400


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

* F. Shen and A. Wang, “Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula,” Appl. Opt., vol. 45, no. 6, pp. 1102–1110, 2006.

* Engineering optics with matlab"

* Fast Bessel and Hankle functions: https://dlmf.nist.gov/10.2#E5

* Beam width: https://en.wikipedia.org/wiki/Beam_diameter

* JA Oglivy "Theory of wave scattering from random surfaces" Adam Hilger 


Credits
---------------------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

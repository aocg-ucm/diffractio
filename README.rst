================================================
Python Diffraction-Interference module
================================================


.. image:: https://img.shields.io/pypi/dm/diffractio
        :target: https://pypi.org/project/diffractio/

.. image:: https://img.shields.io/pypi/v/diffractio.svg
        :target: https://pypi.org/project/diffractio/

.. image:: https://readthedocs.org/projects/diffractio/badge/?version=latest
        :target: https://diffractio.readthedocs.io/en/latest/
        :alt: Documentation Status


* Free software: GPLv3 license

* Documentation: https://diffractio.readthedocs.io/en/latest/


.. image:: logo.png
   :width: 75
   :align: right


Features
----------------------

Diffractio is a Python library for Diffraction and Interference Optics.

It implements Scalar and vector Optics. 

The **scalar** propagation schemes are implemented in modules:

* X - fields are defined in the x axis.
* XY - fields are defined in the xy transversal plane.
* XYZ - fields are defined in the xyz volume.
* XZ - fields are defined in the xz plane, being z the propagation direction.
* Z - fields are defined in the z axis.

Each scheme present three modules:

* sources: Generation of light.
* masks: Masks and Diffractive Optical elements.
* fields:  Propagation techniques, parameters and general functions.

The **vector** propagation schemes are implemented in modules:

* vector_X - fields are defined in the x axis.
* vector_XY - fields are defined in xy transversal plane.
* vector_XYZ - fields are defined in the xyz axis.
* vector_XZ - fields are defined in the xz axis.
* vector_Z - fields are defined in the z axis.

For the vector analysis, we also use the py_pol module: https://pypi.org/project/py-pol/


Sources
========

One main part of this software is the generation of optical fields such as:

* Plane waves.
* Spherical waves.
* Gaussian beams.
* Bessel beams.
* Vortex beams.
* Laguerre beams.
* Hermite-Gauss beams.
* Zernike beams...

.. image:: readme1.png
   :width: 400


Masks
=============

Another important part of Diffractio is the generation of masks and Diffractive Optical Elements such as:

* Slits, double slits, circle, square, ring ...
* Lenses, diffractive lenses, aspherical lenses...
* Axicon, prisms, biprism, image, rough surface, gray scale ...
* Gratings: Ronchi, phase, radial, angular, sine, forked, blazed, 2D, chess...
* Others: Masks can also be loaded (.png, .dxf for 2D and .stl for 3D).

.. image:: readme2.png
   :height: 400

In the XZ scheme there is also a number of masks:

* image, cylinder, layer, lenses, wedge, prism, probe, gratings...
* Additionally, any X_mask can be extruded to the XZ scheme.
* Masks from functions.
* Surfaces can be added to objects.
* Others: Masks can also be loaded (.png, .dxf)

.. image:: readme3.png
   :height: 400


Scalar Fields
==================

Several propagation algorithms for propagation of light are implemented in the *fields* modules:

The main algorithms for Scalar propagation are:


* **Fast Fourier Transform (FFT)** which allows, in a single step, to determine the field at the far field.

* **Rayleigh-Sommerfeld (RS)** [Appl. Opt., 45(6) 1102–1110, (2006)] RS allows, in a single step, to propagate to a near or far observation plane, which allows fast computations. The fields and the masks must be defined in a plane.

* **Plane Wave Decomposition (PWD)** [Opt. Comm. 281 4219-4233 (2008)] PWD allows to propagate to a near or far observation plane. It presents a complexity of O(n) in the two-dimensional and O(n2) in the three-dimensional case. It is computed according to the split-step propagation scheme.

* **Beam propagation method (BPM)** [Appl. Opt. 24, 3390-3998 (1978)] which allows to analyze the propation of light in volumetric elements, such as spheres, cylinders and other complex forms, provided that the spatial variations in the refractive index are small. It allows graded index structures. It presents a complexity of O(n) in the two-dimensional and O(n2) in the three-dimensional case. It is computed according to the split-step propagation scheme.

* **Wave Propagation Method (WPM)**. [Appl. Opt. 32, 4984 (1993)] WPM was introduced in order to overcome the major limitations of the beam propagation method (BPM). With the WPM, the range of application can be extended from the simulation of waveguides to simulation of other optical elements like lenses, prisms and gratings. WPM can accurately simulate scalar light propagation in inhomogeneous media at high numerical apertures, and provides valid results for propagation angles up to 85° and that it is not limited to small index variations in the axis of propagation. Fast implementation with discrete number of refractive indexes is also implemented.

* **Chirped Z-Transform (CZT)**.  [Light: Science and Applications, 9(1), (2020)] CZT allows, in a single step, to propagate to a near or far observation plane. It present advantages with respecto to RS algorithm, since the region of interest and the sampling numbers can be arbitrarily chosen, endowing the proposed method with superior flexibility. CZT algorithm allows to have a XY mask and compute in XY, Z, XZ, XYZ schemes, simply defining the output arrays.



When possible, multiprocessing is implemented for a faster computation.

The fields, masks, and sources can be stored in files.

Also drawings can be easily obtained, for intensity, phase, fields, etc.

In some modules, videos can be generated for a better analysis of optical fields.

.. image:: readme6.png
   :width: 600


The main algorithms for Vector propagation are:

* **Vector Fast Fourier Tranform (VFFT)**, which allows to determine the (Ex, Ey, Ez) fields at the far field.

* **Vector Rayleigh-Sommerfeld (VRS)**. The VRS method [Laser Phys. Lett. 10(6) 065004 (2013)] allows to propagate (Ex,Ey,Ez) fields offering the advantage of significant reduction in computation, from flat diffractive elements (Thin Element Approximation) with full control of polarization. It addresses simultaneously both longitudinal polarization. This approach offers the advantage of significant reduction in computation.

* **Vector Chirp Z-Transform (VCZT)**.  [Light: Science and Applications, 9(1), (2020)]. CZT is also implemented in vector fields.

* **Fast Polarized Wave Propagation Method (FPWPM)**  [Opt Express. 30(22) 40161-40173 (2022)]  Wave Propagation Method for vector fields. It is an efficient method for vector wave optical simulations of microoptics. The FPWPM is capable of handling comparably large simulation volumes while maintaining quick runtime.  By considering polarization in simulations, the FPWPM facilitates the analysis of optical elements which employ this property of electromagnetic waves as a feature in their optical design, e.g., diffractive elements, gratings, or optics with high angle of incidence like high numerical aperture lenses.


Vector beams
----------------------

Here, we implement new classes where the E_x, E_y, and E_z fields are generated and propagated using Rayleigh-Sommerfeld and Chirped z-transform algorithms.
Also, simple and complex polarizing masks can be created.

**Intensity of vector field**

.. image:: readme4.png
   :width: 700

**Polarization: Stokes parameters**

.. image:: readme5.png
   :width: 700


Other features
=================

* Intensity, MTF and other parameters are obtained from the optical fields.

* Fields can be added simply with the + signe, and interference is produced. Masks can be multiplied, added and substracted in order to make complex structures

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



Collaborators
---------------------------

* Ángela Soria Garcia

* Jesús del Hoyo Muñoz

* Francisco Jose Torcal-Milla



Citing
---------------------------

There is a `paper about Diffractio <https://doi.org/10.1117/12.3021879>`_.

If you are using Diffractio in your scientific research, please help our scientific visibility by citing our work.


   Luis Miguel Sanchez-Brea, Angela Soria-Garcia, Joaquin Andres-Porras, Veronica Pastor-Villarrubia, Mahmoud H. Elshorbagy, Jesus del Hoyo Muñoz, Francisco Jose Torcal-Milla, and Javier Alda "Diffractio: an open-source library for diffraction and interference calculations", Proc. SPIE 12997, Optics and Photonics for Advanced Dimensional Metrology III, 129971B (18 June 2024); https://doi.org/10.1117/12.3021879 


BibTex:

.. code::

   @inproceedings{10.1117/12.3021879,
      author = {Luis Miguel Sanchez-Brea and Angela Soria-Garcia and Joaquin Andres-Porras and Veronica Pastor-Villarrubia and Mahmoud H. Elshorbagy and Jesus del Hoyo Mu{\~n}oz and Francisco Jose Torcal-Milla and Javier Alda},
      title = {{Diffractio: an open-source library for diffraction and interference calculations}},
      volume = {12997},
      booktitle = {Optics and Photonics for Advanced Dimensional Metrology III},
      editor = {Peter J. de Groot and Felipe Guzman and Pascal Picart},
      organization = {International Society for Optics and Photonics},
      publisher = {SPIE},
      pages = {129971B},
      keywords = {Design of micro-optical devices, Diffractive optical elements, Propagation algorithms, Scalar propagation, Vector propagation},
      year = {2024},
      doi = {10.1117/12.3021879},
      URL = {https://doi.org/10.1117/12.3021879}
   }





Scalar algorithms
---------------------------


**RS**

* Shen, F. & Wang, A. "Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula," Appl. Opt. 45, 1102–1110 (2006).

**PWD**

* Kozacki, T. "Numerical errors of diffraction computing using plane wave spectrum decomposition," Opt. Comm. 281 4219-4233 (2008).

**CZT**

* Bluestein, L., "A linear filtering approach to the computation of the discrete Fourier transform," Northeast Electronics Research and Engineering Meeting Record 10, 218-219 (1968).

* Hu Y. et al. "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method" Light: Science & Applications  9(119) (2020)

**WPM**

* Brenner K.H., Singer W. , “Light propagation through micro lenses: a new simulation method”, Appl. Opt., 32(6) 4984-4988 (1993).

* Schmidt S. et al. "Wave-optical modeling beyond the thin-element-approximation" Opt. Express 24, 30188 (2016).

* Brenner K.H. "A high-speed version of the wave propagation method applied to micro-optics."  16th Workshop on Information Optics (WIO). IEEE (2017)

* Schmidt S. et al. "Rotationally symmetric formulation of the wave propagation method-application to the straylight analysis of diffractive lenses" Opt. Lett. 42, 1612 (2017).


Scalar algorithms
---------------------------

**VFFT** 

Kornél J. and Bokor N., 2010. “Intensity Control of the Focal Spot by Vectorial Beam Shaping.” Optics Communications 283 (24): 4859–65. https://doi.org/10.1016/j.optcom.2010.07.030.

**VRS**

* Ye, H. et al. "Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: Vectorial Rayleigh-Sommerfeld method" Laser Phys. Lett. 10, (2013).

**VCZT**

* Leutenegger M. et al. "Fast focus field calculations" Optics Express 14(23) 11277 (2006).

* Hu Y. et al. "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method" Light: Science & Applications  9(119) (2020)

**FPWPM** 

* Wende M,et al. "Fast algorithm for the simulation of 3D-printed microoptics based on the vector wave propagation method". Opt Express. 30(22) 40161-40173 (2022)


Other References
---------------------------

* J.W. Goodman, "Introduction to Fourier Optics" McGraw-Hill, 1996.

* B.E. Saleh y M. C. Teich, "Fundamentals of photonics" John Wiley & Sons, 2019.

* Z.Qiwen, "Vectorial optical fields: Fundamentals and applications" World scientific, 2013.

* "Numerical Methods in Photonics Lecture Notes".  http://ecee.colorado.edu/~mcleod/teaching/nmip/lecturenotes.html.


Credits
---------------------------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

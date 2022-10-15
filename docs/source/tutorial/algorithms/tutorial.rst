Algorithms
=============

Propagation algorithms is a fundamental key in Diffractio package. The algorithms implmented are:

**Scalar Propagation**

* Fast Fourier Transform (FFT).
* Rayleigh Sommerfeld (RS).
* Plane Wave Descomposition (PWD).
* Beam Propagation Method (BPM).
* Wave Propagation Method (WPM).
* Chirp z-transform (CZT).

**Vector propagation**

* Vector Fast Fourier Tranform (VFFT).
* Vector Rayleigh-Sommerfeld (VRS).
* Vector Chirped z-transform (VCZT)

**Fast Fourier Transform (FFT)**

It allows, in a single step to determine the field at the far field.
The fields and the masks must be defined in a plane.


**Rayleigh-Sommerfeld (RS)**
- Shen, F., and Wang, A. (2006). Fast-Fourier-transform based numerical integration method for the Rayleigh-Sommerfeld diffraction formula. Applied Optics, 45(6), 1102–1110. https://doi.org/10.1364/AO.45.001102

Single step to propagate to a near or far observation plane, which allows fast computations. 


**Beam propagation method (BPM)** 

- Feit, M. D., and Fleck, J. A. (1978). Light propagation in graded-index optical fibers.

Propation of light in volumetric elements, provided that the spatial variations in the refraction index are small. It allows graded index structures. It presents a complexity of O(n) in the two-dimensional and O(n2) in the three-dimensional case. Computed according to split-step propagation scheme.


**Wave Propagation Method (WPM)**

- Brenner, K.-H. H., and Singer, W. (1993). Light propagation through microlenses: a new simulation method. Applied Optics, 32(26), 4984–4988. https://doi.org/10.1364/ao.32.004984

- Brenner, K. H. (2017). A high-speed version of the wave propagation method applied to micro-optics. 2017 16th Workshop on Information Optics, WIO 2017, 1, 2–4. https://doi.org/10.1109/WIO.2017.8038108

- Schmidt, S., Thiele, S., Herkommer, A., Tünnermann, A., and Gross, H. (2017). Rotationally symmetric formulation of the wave propagation method-application to the straylight analysis of diffractive lenses. Optics Letters, 42(8), 1612. https://doi.org/10.1364/ol.42.001612

Solves the major limitations of the beam propagation method (BPM). The wave propagation scheme provides valid results for propagation angles up to 85° and that it is not limited to small index variations in the axis of propagation. Very fast with a discrete number of refraction indexes.


**Chirped z Transform (CZT)**

- Leutenegger, M., Rao, R., Leitgeb, R. A., and Lasser, T. (2006). Fast focus field calculations. Optics Express, 14(23), 11277–11291. http://lob.epfl.ch/

- Hu, Y., Wang, Z., Wang, X., Ji, S., Zhang, C., Li, J., Zhu, W., Wu, D., and Chu, J. (2020). Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. Light: Science and Applications, 9(1). https://doi.org/10.1038/s41377-020-00362-z


**Vector Fast Fourier Transform (VFFT)**

- todo


**Vector Rayleigh-Sommerfeld (VRS)**

- Ye, H., Qiu, C. W., Huang, K., Teng, J., Luk’Yanchuk, B., and Yeo, S. P. (2013). Creation of a longitudinally polarized subwavelength hotspot with an ultra-thin planar lens: Vectorial Rayleigh-Sommerfeld method. Laser Physics Letters, 10(6). https://doi.org/10.1088/1612-2011/10/6/065004

The VRS method propagates (Ex,Ey,Ez) fields offering the advantage of significant reduction in computation, from flat diffractive elements (Thin Element Approximation) with full control of polarization. It addresses simultaneously both longitudinal polarization. This approach offers the advantage of significant reduction in computation.


**Vector Chirped Z Transform (VCZT)** 

- Hu, Y., Wang, Z., Wang, X., Ji, S., Zhang, C., Li, J., Zhu, W., Wu, D., and Chu, J. (2020). Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. Light: Science and Applications, 9(1). https://doi.org/10.1038/s41377-020-00362-z


.. toctree::
   :maxdepth: 4
   :numbered:
   :glob:


   FFT.ipynb
   RS.ipynb
   BPM.ipynb
   WPM.ipynb
   CZT.ipynb
   VFFT.ipynb
   VRS.ipynb
   VCZT.ipynb
   comparison/RS_vs_WPM.ipynb
   comparison/WPM_vs_BPM.ipynb
   comparison/CZT_vs_RS.ipynb
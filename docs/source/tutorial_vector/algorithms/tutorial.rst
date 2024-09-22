Algorithms
=============

The algorithms implemented for vector propagation are:

* Vector Fast Fourier Tranform (VFFT).
* Vector Rayleigh-Sommerfeld (VRS).
* Vector Chirped z-transform (VCZT)
* 

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
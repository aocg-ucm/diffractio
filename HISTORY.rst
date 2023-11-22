=======
History
=======

0.0.0 (2017-01-01)
------------------------

Pre-alpha state.

* I have been developing and using the module diffractio for at least 5 years for teaching and research. It was in python2 version and not completely clear.


0.0.1 (2019-02-09)
------------------------

Pre-alpha state.

* Modules are passed to python3.


0.0.2 (2019-10-01)
------------------------
Alpha state

* copyreg removed (previous not worked)
* change fft to scipy: "from scipy.fftpack import fft, ifft, fftshift"


  First release on PyPI in pre-alpha state.



0.0.5 (2019-10-06)
------------------------
Alpha state

* Included vector (fields, sources, masks) modules, tests, and tutorial.

0.0.6 (2019-10-22)
------------------------
Alpha state

* Finished first version of examples and tutorial.


0.0.7 and 0.0.8 (2020-05-02)
--------------------------------
Alpha state

* convert_mask -> extrude_mask
* Improved documentation
* Implemented PWD
* Reduced size of html


0.0.9 and 0.0.10 (2020-05-02)
--------------------------------
Alpha state

* Improvement to documentation (readthedocs)

0.0.11 and 0.0.12 (2021-05-09)
--------------------------------
Alpha state

* Solved big errata in vector_paraxial_masks
* Solved errata in XY: kernelRS and kernelRSinverse
* Improved documentation, tutorial and examples of vector_paraxial
* Scalar_mask_XY: dots
* Scalar_mask_X: dots
* change save_data in all classes to simplify
* Changes in docs to include WPM
* Execution tests

0.0.13 (2021-08-11)
--------------------------------
Alpha state

* wpm bpm 3d, wpm without storing,  xy search focus
* fix bug 2D gratings
* fix bug aspheric X and new aspheric XY


0.1.0 (2022-10-12)
--------------------------------
Beta state


* Fix bugs:

  - radial and azimuthal vector waves
  - Change in criterion of plane waves to Physics (ISO 80000-2:2019 convention): https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
  - constant_wave -> constant_polarization in vector_sources_XY
  - mask_circle -> pupil
  - 

* New vector_fields_XY:

  - vector_paraxial_fields -> vector_fields

* New schemes mainly for representation:

  - Scalar_field_Z
  - vector_X, vector_Z, vector_XZ, vector_XYZ


* New Scalar_mask_XY: 

  - angular_aperture 
  - edge_series 
  - slit_series 
  - rings


* New propagation algorithms:

  - WPM and BPM in 3D
  - WPM and BPM in 3D without storing intermediate planes
  - VFFT (vector FFT and inverse FFT)
  - VRS (vector Rayleigh-Sommerfeld propagation, with Ez field)
  - Chirped z-transform (CZT) for scalar fields
  - Chirped z-transform (CZT) for vector fields


* Other:

  - Pupil function in XY and vector XY
  - Remove mask parameter from lenses. This may produce incompatibilities with former code
  - Improving drawings

0.1.1 (2022-10-16)
--------------------------------
* Vector fields are not longer paraxial.

  - The propagation algorithms implemented (VRS, VFFT and VCZT) provide :E_z: field. This allows to analyze longitudinal fields.The modules and classes elliminate changes their name. For example vector_paraxial_fields_X is now vector_fields_X.


* New propagation algorithm Chirped Z-Transform (CZT) is avaliable for X and XY fields.

  - This algorithms produce similar results to RS and VRS fields, but there are significant advantages:

  - The output field is not necessarily the same as the input field. This is important, for example, when analyzing the focus of a lens, since the computing time is dedicated to areas with light.

  - The output field does not necessarily have the same dimensionality of the input field. For example, when the mask is XY, we can have the data only at (x=0, y=0, z) axis.

  - Acceleration in computing and reduction of memory usage.


* New modules for visualization and data analysis are provided.

  - Scalar_field_Z can be used, for example, to analysis of intensity light at axis.

  - Vector_fields_Z, Vector_fields_XZ, and Vector_fields_XYZ have been developed, as VCZT algorithm can provide these data.

* Plane Wave Descomposition algorithm (PWD) is deprecated.

* Some importante bugs have been solved. For example the definition of the spherical coordinates in some sources (which not used standard physics criterion).

* Mask parameters is removed in some XY masks, as lenses, FPZ, etc. The new way to do this is the .pupil() function.

* Smooth_refraction index can be used also for Wave Propagation Method algorithm (WPM).


0.2.2 (2023-11-21)
--------------------------------

* Improvement:

  - XYZ drawings is removed temporally.

  - utils_slicer.py is deprecated.

  -  

* Fix bugs:

  - Bug in XYZ video

  - Blazed grating wrong defined. 
  
  - bug in Scalar_field_XY.kernelRSinverse

* New features:

  - XY masks: new polygon, regular_polygon, star, and superformula functions. 

  - XY masks: new circular_sector function.

  - XY masks: new lens_cyliindrical function.

  - X, XY, XYZ fields: new conjugate function.

  - WPM function without storing all the data.

  - X fields: inverse_amplitude, inverse_phase, pupil


Future
--------------------------------


  - Implement plotly for drawings

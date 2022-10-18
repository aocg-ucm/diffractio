================================================
New Features
================================================

In versión 0.1.1, serveral new features are available:

* Vector fields are not longer paraxial.

   The propagation algorithms implemented (VRS, VFFT and VCZT) provide :E_z: field. This allows to analyze longitudinal fields.

   As a consequence the modules and classes elliminate changes their name. For example vector_paraxial_fields_X is now vector_fields_X.


* New propagation algorithm Chirped Z-Transform (CZT) is avaliable for X and XY fields.

   This algorithms produce similar results to RS and VRS fields, but there are significant advantages:

   - The output field is not necessarily the same as the input field. This is important, for example, when analyzing the focus of a lens, since the computing time is dedicated to areas with light.

   - The output field does not necessarily have the same dimensionality of the input field. For example, when the mask is XY, we can have the data only at (x=0, y=0, z) axis.

   - As a consequence the acceleration of the field computation can be accelerated as much as x100.

   - Another significant advantage is the reduction of memory usage.


* New modules for visualization and data analysis are provided.

   - Scalar_field_Z can be used, for example, to analysis of intensity light at axis.

   - Vector_fields_Z, Vector_fields_XZ, and Vector_fields_XYZ have been developed, as VCZT algorithm can provide these data.

   
* Plane Wave Descomposition algorithm (PWD) is deprecated as Rayleigh-Sommerfeld algorithm (RS) produces better results.

* Some importante bugs have been solved. For example the definition of the spherical coordinates in some sources (which not used standard physics criterion).

* Mask parameters is removed in some XY masks, as lenses, FPZ, etc. The new way to do this is the .pupil() function.

* Smooth_refraction index can be used also for Wave Propagation Method algorithm (WPM).


Warning: Some of this features make your code slightly change.
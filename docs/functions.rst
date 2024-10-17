Functions:
================================

__init__.py
__________________________________________________





config.py
__________________________________________________





scalar_fields_X.py
__________________________________________________

 Class: **Scalar_field_X**. (44 functions)
    - CZT

    - MTF

    - PWD_kernel

    - RS

    - WPM

    - WPM_schmidt_kernel

    - _RS_

    - __add__

    - __init__

    - __mul__

    - __str__

    - __sub__

    - add

    - average_intensity

    - clear_field

    - conjugate

    - cut_resample

    - draw

    - duplicate

    - extended_polychromatic_source

    - extended_source_multiprocessing

    - fft

    - filter

    - get_RS_minimum_z

    - get_RS_minimum_z

    - get_edges

    - ifft

    - incident_field

    - insert_array_masks

    - insert_mask

    - intensity

    - inverse_amplitude

    - inverse_phase

    - kernelRS

    - kernelRSinverse

    - load_data

    - normalize

    - oversampling

    - polychromatic_multiprocessing

    - pupil

    - quality_factor

    - reduce_to_1

    - repeat_structure

    - save_data





scalar_fields_XY.py
__________________________________________________

 Class: **Scalar_field_XY**. (58 functions)
    - CZT

    - MTF

    - PWD_kernel

    - RS

    - WPM

    - WPM_schmidt_kernel

    - _RS_

    - __add__

    - __drawAmplitude__

    - __draw_field__

    - __draw_intensity__

    - __draw_phase__

    - __draw_real_field__

    - __init__

    - __mul__

    - __rotate__

    - __str__

    - __sub__

    - add

    - animate

    - apodization

    - average_intensity

    - beam_width_4s

    - binarize

    - clear_field

    - conjugate

    - cut_resample

    - discretize

    - draw

    - draw_profile

    - duplicate

    - fft

    - fft_proposal

    - get_RS_minimum_z

    - get_RS_minimum_z

    - get_amplitude

    - get_edges

    - get_phase

    - ifft

    - incident_field

    - intensity

    - kernelFresnel

    - kernelRS

    - kernelRSinverse

    - load_data

    - normalize

    - oversampling

    - profile

    - pupil

    - quality_factor

    - reduce_to_1

    - remove_phase

    - rotate

    - save_data

    - save_mask

    - search_focus

    - send_image_screen

    - video





scalar_fields_XYZ.py
__________________________________________________

 Class: **Scalar_field_XYZ**. (28 functions)
    - BPM

    - PWD

    - RS

    - RS_amplification

    - WPM

    - __RS_multiprocessing__

    - __add__

    - __init__

    - __rotate__

    - __rotate_axis__

    - __str__

    - __sub__

    - clear_field

    - clear_refractive_index

    - conjugate

    - cut_resample

    - duplicate

    - final_field

    - incident_field

    - intensity

    - load_data

    - normalize

    - oversampling

    - reduce_to_1

    - save_data

    - to_Scalar_field_XY

    - to_Scalar_field_XZ

    - xy_2_xyz


 Standalone functions: (13 functions)

  - average_intensity

  - beam_widths

  - draw_XY

  - draw_XYZ

  - draw_XZ

  - draw_YZ

  - draw_proposal

  - f

  - surface_detection

  - to_Scalar_field_YZ

  - to_Scalar_field_Z

  - video

  - video_isovalue





scalar_fields_XZ.py
__________________________________________________

 Class: **Scalar_field_XZ**. (50 functions)
    - BPM

    - BPM_back_propagation

    - BPM_inverse

    - BPM_polychromatic

    - PWD

    - RS

    - RS_polychromatic

    - WPM

    - WPM_polychromatic

    - __BPM__

    - __RS_multiprocessing__

    - __add__

    - __init__

    - __rotate__

    - __str__

    - __sub__

    - __update__

    - _detect_transitions_

    - animate

    - average_intensity

    - beam_widths

    - check_intensity

    - clear_field

    - clear_refractive_index

    - cut_resample

    - detect_index_variations

    - draw

    - draw_incident_field

    - draw_profiles_interactive

    - draw_refractive_index

    - duplicate

    - fast_propagation

    - final_field

    - incident_field

    - init

    - intensity

    - load_data

    - mask_field

    - normalize

    - oversampling

    - profile_longitudinal

    - profile_transversal

    - reduce_to_1

    - refractive_index_from_scalar_mask_XY

    - rotate_field

    - save_data

    - search_focus

    - smooth_refractive_index

    - surface_detection

    - video





scalar_fields_Z.py
__________________________________________________

 Class: **Scalar_field_Z**. (17 functions)
    - DOF

    - FWHM1D

    - __add__

    - __init__

    - __str__

    - __sub__

    - average_intensity

    - clear_field

    - conjugate

    - cut_resample

    - draw

    - duplicate

    - intensity

    - load_data

    - normalize

    - oversampling

    - save_data





scalar_masks_X.py
__________________________________________________

 Class: **Scalar_mask_X**. (28 functions)
    - __init__

    - aspheric

    - binary_code

    - binary_code_positions

    - binary_grating

    - biprism_fresnel

    - biprism_fresnel_nh

    - blazed_grating

    - chirped_grating

    - chirped_grating_p

    - chirped_grating_q

    - dots

    - double_slit

    - dust

    - dust_different_sizes

    - filter

    - fresnel_lens

    - gray_scale

    - lens

    - lens_spherical

    - mask_from_array

    - mask_from_function

    - prism

    - ronchi_grating

    - roughness

    - sine_grating

    - slit

    - two_levels





scalar_masks_XY.py
__________________________________________________

 Class: **Scalar_mask_XY**. (75 functions)
    - __init__

    - angular_aperture

    - angular_grating

    - archimedes_spiral

    - area

    - aspheric

    - axicon

    - axicon_binary

    - binary_grating

    - biprism_fresnel

    - blazed_grating

    - circle

    - circle_rough

    - circular_sector

    - cross

    - crossed_slits

    - dots

    - dots_regular

    - double_slit

    - double_slit_rough

    - dxf

    - edge_rough

    - edge_series

    - elliptical_phase

    - extrude_mask_x

    - filter

    - forked_grating

    - fresnel_lens

    - fresnel_lens_rough

    - grating_2D

    - grating_2D_chess

    - gray_scale

    - hammer

    - hermite_gauss_binary

    - hyperbolic_grating

    - image

    - insert_array_masks

    - inverse_amplitude

    - inverse_phase

    - laguerre_gauss_binary

    - laguerre_gauss_spiral

    - lens

    - lens_cylindrical

    - lens_spherical

    - mask_from_function

    - masks_to_positions

    - one_level

    - photon_sieve

    - polygon

    - prism

    - radial_grating

    - regular_polygon

    - repeat_structure

    - ring

    - ring_rough

    - rings

    - ronchi_grating

    - roughness

    - set_amplitude

    - set_phase

    - sine_edge_grating

    - sine_grating

    - sinusoidal_slit

    - slit

    - slit_rough

    - slit_series

    - square

    - square_circle

    - star

    - super_ellipse

    - super_gauss

    - superformula

    - triangle

    - two_levels

    - widen





scalar_masks_XYZ.py
__________________________________________________

 Class: **Scalar_mask_XYZ**. (6 functions)
    - __init__

    - cylinder

    - object_by_surfaces

    - sphere

    - square

    - stl





scalar_masks_XZ.py
__________________________________________________

 Class: **Scalar_mask_XZ**. (31 functions)
    - __init__

    - _discretize_

    - add_surfaces

    - aspheric_lens

    - aspheric_surface_z

    - biprism

    - cylinder

    - discretize_refractive_index

    - dots

    - dxf

    - extrude_mask

    - image

    - layer

    - lens_convergent

    - lens_divergent

    - lens_plane_convergent

    - lens_plane_divergent

    - mask_from_array

    - mask_from_array_proposal

    - mask_from_function

    - object_by_surfaces

    - prism

    - probe

    - rectangle

    - ronchi_grating

    - rough_sheet

    - semi_cylinder

    - semi_plane

    - sine_grating

    - slit

    - wedge





scalar_sources_X.py
__________________________________________________

 Class: **Scalar_source_X**. (8 functions)
    - __init__

    - gauss_beam

    - gauss_beams_several_inclined

    - gauss_beams_several_parallel

    - plane_wave

    - plane_waves_dict

    - plane_waves_several_inclined

    - spherical_wave





scalar_sources_XY.py
__________________________________________________

 Class: **Scalar_source_XY**. (13 functions)
    - __init__

    - bessel_beam

    - gauss_beam

    - gauss_beams_several_inclined

    - gauss_beams_several_parallel

    - hermite_gauss_beam

    - laguerre_beam

    - plane_wave

    - plane_waves_dict

    - plane_waves_several_inclined

    - spherical_wave

    - vortex_beam

    - zernike_beam





utils_common.py
__________________________________________________


 Standalone functions: (14 functions)

  - add

  - check_none

  - clear_all

  - computer_parameters

  - date_in_name

  - decorator

  - get_date

  - load_data_common

  - oversampling

  - print_axis_info

  - print_data_dict

  - save_data_common

  - several_propagations

  - wrapper





utils_drawing.py
__________________________________________________


 Standalone functions: (11 functions)

  - change_image_size

  - concatenate_drawings

  - draw2D

  - draw_several_fields

  - extract_image_from_video

  - make_video_from_file

  - normalize_draw

  - prepare_drawing

  - prepare_video

  - reduce_matrix_size

  - view_image





utils_drawing3D.py
__________________________________________________


 Standalone functions: (5 functions)

  - draw

  - load_stl

  - show_stl

  - video_isovalue

  - voxelize_volume_diffractio





utils_dxf.py
__________________________________________________


 Standalone functions: (4 functions)

  - binarize

  - load_dxf

  - set_pixel_density

  - set_pixel_size





utils_math.py
__________________________________________________


 Standalone functions: (36 functions)

  - Bluestein_dft_x

  - Bluestein_dft_xy

  - amplitude2phase

  - binarize

  - cart2pol

  - curl

  - cut_function

  - delta_kronecker

  - discretize

  - distance

  - divergence

  - dot_product

  - fZernike

  - fft_convolution1d

  - fft_convolution2d

  - fft_correlation1d

  - fft_correlation2d

  - fft_filter

  - filter_edge_1D

  - filter_edge_2D

  - find_extrema

  - find_local_extrema

  - get_amplitude

  - get_edges

  - get_k

  - get_phase

  - laguerre_polynomial_nk

  - nearest

  - nearest2

  - nextpow2

  - normalize

  - phase2amplitude

  - pol2cart

  - reduce_to_1

  - rotate_image

  - vector_product





utils_multiprocessing.py
__________________________________________________

 Class: **auxiliar_multiprocessing**. (6 functions)
    - __function_process__

    - __init__

    - creation_dictionary_multiprocessing

    - execute_multiprocessing

    - execute_multiprocessing

    - method_single_proc


 Standalone functions: (3 functions)

  - _pickle_method

  - _unpickle_method

  - separate_from_iterable





utils_optics.py
__________________________________________________


 Standalone functions: (24 functions)

  - DOF

  - FWHM1D

  - FWHM2D

  - MTF_ideal

  - MTF_parameters

  - beam_width_1D

  - beam_width_2D

  - convert_amplitude2heigths

  - convert_phase2heigths

  - detect_intensity_range

  - field_parameters

  - fresnel_equations

  - fresnel_equations_kx

  - gauss_spectrum

  - lines_mm_2_cycles_degree

  - lorentz_spectrum

  - normalize_field

  - refractive_index

  - roughness_1D

  - roughness_2D

  - transmitances_reflectances

  - transmitances_reflectances_kx

  - uniform_spectrum

  - width_percentage





utils_tests.py
__________________________________________________


 Standalone functions: (9 functions)

  - _test_slit_RS_XY

  - benchmark_num_pixels

  - benchmark_processors_n_pixels

  - comparison

  - ejecute_multiprocessing

  - run_benchmark

  - save_data_test

  - save_figure_test

  - test_pixels





utils_typing.py
__________________________________________________


 Standalone functions: (2 functions)

  - make_new_user

  - stop





vector_fields_X.py
__________________________________________________

 Class: **Vector_field_X**. (21 functions)
    - __add__

    - __draw1__

    - __draw_fields__

    - __draw_intensities__

    - __draw_intensity__

    - __draw_param_ellipse__

    - __draw_phases__

    - __draw_stokes__

    - __init__

    - __str__

    - apply_mask

    - clear_field

    - draw

    - duplicate

    - get

    - intensity

    - load_data

    - normalize

    - polarization_ellipse

    - polarization_states

    - save_data





vector_fields_XY.py
__________________________________________________

 Class: **Vector_field_XY**. (33 functions)
    - IVFFT

    - RS

    - VCZT

    - VFFT

    - VRS

    - __add__

    - __draw1__

    - __draw_ellipses__

    - __draw_fields__

    - __draw_intensities__

    - __draw_intensities_rz__

    - __draw_intensity__

    - __draw_param_ellipse__

    - __draw_phases__

    - __draw_stokes__

    - __init__

    - __rotate__

    - __str__

    - _compute1Elipse__

    - apply_mask

    - clear

    - cut_resample

    - draw

    - duplicate

    - get

    - intensity

    - load_data

    - normalize

    - polarization_ellipse

    - polarization_states

    - pupil

    - save_data

    - to_py_pol





vector_fields_XYZ.py
__________________________________________________

 Class: **Vector_field_XYZ**. (13 functions)
    - __add__

    - __init__

    - __str__

    - clear_field

    - duplicate

    - get

    - intensity

    - load_data

    - normalize

    - polarization_ellipse

    - polarization_states

    - save_data

    - to_Vector_field_XY


 Standalone functions: (7 functions)

  - _compute1Elipse__

  - draw_XY

  - draw_XZ

  - draw_YZ

  - to_Vector_field_XZ

  - to_Vector_field_YZ

  - to_Vector_field_Z





vector_fields_XZ.py
__________________________________________________

 Class: **Vector_field_XZ**. (39 functions)
    - FP_PWD_kernel_simple

    - FP_WPM

    - FP_WPM_schmidt_kernel

    - Poynting_total

    - Poynting_vector

    - Poynting_vector_averaged

    - __add__

    - __draw1__

    - __draw_EH__

    - __draw_ellipses__

    - __draw_fields__

    - __draw_intensities__

    - __draw_intensity__

    - __draw_param_ellipse__

    - __draw_phases__

    - __draw_stokes__

    - __init__

    - __str__

    - apply_mask

    - check_energy

    - clear_field

    - draw

    - draw2D_xz

    - draw_field

    - duplicate

    - energy_density

    - final_field

    - format_drawing

    - get

    - incident_field

    - intensity

    - irradiance

    - load_data

    - normalize

    - polarization_ellipse

    - polarization_ellipse

    - polarization_states

    - refractive_index_from_scalarXZ

    - save_data





vector_fields_Z.py
__________________________________________________

 Class: **Vector_field_Z**. (21 functions)
    - __add__

    - __draw1__

    - __draw_fields__

    - __draw_intensities__

    - __draw_intensity__

    - __draw_param_ellipse__

    - __draw_phases__

    - __draw_stokes__

    - __init__

    - __str__

    - apply_mask

    - clear_field

    - draw

    - duplicate

    - get

    - intensity

    - load_data

    - normalize

    - polarization_ellipse

    - polarization_states

    - save_data





vector_masks_XY.py
__________________________________________________

 Class: **Vector_mask_XY**. (18 functions)
    - SLM

    - __add__

    - __init__

    - __mul__

    - __rmul__

    - apply_circle

    - complementary_masks

    - draw

    - duplicate

    - from_py_pol

    - half_waveplate

    - multilevel_mask

    - polarizer_linear

    - polarizer_retarder

    - pupil

    - quarter_waveplate

    - rotation_matrix_Jones

    - scalar_to_vector_mask





vector_sources_XY.py
__________________________________________________

 Class: **Vector_source_XY**. (11 functions)
    - __init__

    - azimuthal_inverse_wave

    - azimuthal_wave

    - constant_polarization

    - define_initial_field

    - local_polarized_vector_wave

    - local_polarized_vector_wave_hybrid

    - local_polarized_vector_wave_radial

    - radial_inverse_wave

    - radial_wave

    - spiral_polarized_beam





Summary
============================


**vector_sources_XY.py**

  Number of lines: 427

  Number of classes: 1

    Class: Vector_source_XY, Number of functions: 11


**scalar_masks_XY.py**

  Number of lines: 2627

  Number of classes: 1

    Class: Scalar_mask_XY, Number of functions: 75


**scalar_fields_XYZ.py**

  Number of lines: 1561

  Number of classes: 3

    Class: Scalar_field_XYZ, Number of functions: 28


**utils_common.py**

  Number of lines: 469


**scalar_fields_Z.py**

  Number of lines: 472

  Number of classes: 1

    Class: Scalar_field_Z, Number of functions: 17


**vector_fields_Z.py**

  Number of lines: 774

  Number of classes: 1

    Class: Vector_field_Z, Number of functions: 21


**scalar_sources_X.py**

  Number of lines: 222

  Number of classes: 1

    Class: Scalar_source_X, Number of functions: 8


**scalar_sources_XY.py**

  Number of lines: 563

  Number of classes: 1

    Class: Scalar_source_XY, Number of functions: 13


**vector_fields_X.py**

  Number of lines: 806

  Number of classes: 1

    Class: Vector_field_X, Number of functions: 21


**utils_drawing3D.py**

  Number of lines: 535


**__init__.py**

  Number of lines: 83


**utils_typing.py**

  Number of lines: 58


**scalar_fields_X.py**

  Number of lines: 1723

  Number of classes: 1

    Class: Scalar_field_X, Number of functions: 44


**config.py**

  Number of lines: 52


**utils_multiprocessing.py**

  Number of lines: 166

  Number of classes: 1

    Class: auxiliar_multiprocessing, Number of functions: 6


**scalar_masks_XYZ.py**

  Number of lines: 261

  Number of classes: 1

    Class: Scalar_mask_XYZ, Number of functions: 6


**utils_optics.py**

  Number of lines: 1228


**vector_fields_XY.py**

  Number of lines: 1996

  Number of classes: 1

    Class: Vector_field_XY, Number of functions: 33


**utils_dxf.py**

  Number of lines: 187


**vector_fields_XZ.py**

  Number of lines: 1892

  Number of classes: 1

    Class: Vector_field_XZ, Number of functions: 39


**utils_tests.py**

  Number of lines: 205


**utils_math.py**

  Number of lines: 981

  Number of classes: 1


**scalar_fields_XY.py**

  Number of lines: 2786

  Number of classes: 1

    Class: Scalar_field_XY, Number of functions: 58


**scalar_fields_XZ.py**

  Number of lines: 2126

  Number of classes: 1

    Class: Scalar_field_XZ, Number of functions: 50


**vector_fields_XYZ.py**

  Number of lines: 690

  Number of classes: 4

    Class: Vector_field_XYZ, Number of functions: 13


**scalar_masks_XZ.py**

  Number of lines: 1626

  Number of classes: 1

    Class: Scalar_mask_XZ, Number of functions: 31


**vector_masks_XY.py**

  Number of lines: 652

  Number of classes: 1

    Class: Vector_mask_XY, Number of functions: 18


**scalar_masks_X.py**

  Number of lines: 978

  Number of classes: 1

    Class: Scalar_mask_X, Number of functions: 28


**utils_drawing.py**

  Number of lines: 452

Total
============================

 Total number of Python files: 29

 Total number of functions: 648

 Total number of lines across all files: 26598


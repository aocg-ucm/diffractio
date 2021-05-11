# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

from . import mm, nm, np, plt, sp, um
from .scalar_masks_XY import Scalar_mask_XY
from .scalar_sources_XY import Scalar_source_XY


def GS_algorithm(source, target, num_steps=11):
    """
    # TODO:
    """

    errors = np.zeros(num_steps)

    z = 1 * mm
    x = target.x
    y = target.y
    num_x = len(x)
    num_y = len(y)
    wavelength = target.wavelength

    radius_x = (x.max() - x.min()) / 2
    radius_y = (y.max() - y.min()) / 2
    x_mean = (x.max() + x.min()) / 2
    y_mean = (y.max() + y.min()) / 2

    circle = Scalar_mask_XY(x, y, wavelength)

    if source is None:
        source = Scalar_source_XY(x, y, wavelength)
        source.plane_wave()

    DOE = Scalar_mask_XY(x, y, wavelength)
    far_field = Scalar_mask_XY(x, y, wavelength)

    target_abs = np.fft.fftshift(np.abs(target.u))
    far_field.u = target_abs * np.exp(
        1j * 2 * np.pi * np.random.rand(num_y, num_x))
    I_result = target_abs**2
    I_result_mean = I_result.mean()

    for i in range(num_steps):
        print("{}/{}".format(i, num_steps), end='\r')
        DOE = far_field.ifft(z=z, shift=False)
        mask = np.angle(DOE.u)
        DOE.u = np.exp(1j * mask)
        field_z = DOE.fft(z=z, shift=False, matrix=True)
        I_z = np.abs(field_z)**2
        I_z = I_z * I_result_mean / I_z.mean()
        far_field.u = target_abs * np.exp(1j * np.angle(field_z))

        errors[i] = compute_error(I_result, I_z)

    if False:
        plt.figure()
        plt.imshow(I_result)
        plt.colorbar()
        plt.figure()
        plt.imshow(I_z)
        plt.colorbar()

    plt.plot(errors, 'k', lw=2)
    plt.title('errors')

    mask = np.fft.fftshift(mask)
    mask = (mask + np.pi) / (2 * np.pi)

    mask_final = Scalar_mask_XY(x, y, wavelength)
    mask_final.u = mask

    return mask_final, errors



def GS_Fresnel_algorithm(source, target, z, num_steps=5, has_draw=False):
    """
    # TODO:
    """

    errors = np.zeros(num_steps)

    x = target.x
    y = target.y
    wavelength = target.wavelength
    num_x = len(x)
    num_y = len(y)

    radius_x = (x.max() - x.min()) / 2
    radius_y = (y.max() - y.min()) / 2
    x_mean = (x.max() + x.min()) / 2
    y_mean = (y.max() + y.min()) / 2

    circle = Scalar_mask_XY(x, y, wavelength)
    circle.circle(r0=(x_mean, y_mean), radius=(radius_x, radius_y))

    if source is None:
        source = Scalar_source_XY(x, y, wavelength)
        source.plane_wave()

    DOE = Scalar_mask_XY(x, y, wavelength)
    field_z = Scalar_mask_XY(x, y, wavelength)
    # DOE.u=np.abs(target.u)*np.exp(1j*2*np.pi*np.random.rand(num_x,num_y))

    u_target = np.abs(target.u)
    I_result = np.abs(target.u)**2
    I_result_mean = I_result.mean()

    field_z.u = u_target * np.exp(2j * np.pi * np.random.rand(num_y, num_x))

    for i in range(num_steps):
        print("{}/{}".format(i, num_steps), end='\r')
        DOE = field_z.RS(z=-z, new_field=True)
        mask = np.angle(DOE.u)
        DOE.u = np.exp(1j * mask)
        field_z = (source * DOE).RS(z=z, new_field=True)
        I_z = np.abs(field_z.u)**2
        I_z = I_z * I_result_mean / I_z.mean()

        field_z.u = u_target * np.exp(1j * np.angle(field_z.u))

        # if has_mask:
        #     DOE.u=np.exp(1j*np.pi*mask)*circle.u

        errors[i] = compute_error(I_result, I_z)

    mask = (mask + np.pi) / (2 * np.pi)
    print(mask.max(), mask.min())

    plt.plot(errors, 'k', lw=2)
    plt.title('errors')

    mask_final = Scalar_mask_XY(x, y, wavelength)
    mask_final.u = mask

    return mask_final, errors



def Wyrowski_algorithm(source, target, num_steps=11):
    """
    # Para Ángela:
    """

    errors = np.zeros(num_steps)

    z = 1 * mm
    x = target.x
    y = target.y
    num_x = len(x)
    num_y = len(y)
    wavelength = target.wavelength

    radius_x = (x.max() - x.min()) / 2
    radius_y = (y.max() - y.min()) / 2
    x_mean = (x.max() + x.min()) / 2
    y_mean = (y.max() + y.min()) / 2

    circle = Scalar_mask_XY(x, y, wavelength)

    if source is None:
        source = Scalar_source_XY(x, y, wavelength)
        source.plane_wave()

    DOE = Scalar_mask_XY(x, y, wavelength)
    far_field = Scalar_mask_XY(x, y, wavelength)

    target_abs = np.fft.fftshift(np.abs(target.u))
    far_field.u = target_abs * np.exp(
        1j * 2 * np.pi * np.random.rand(num_y, num_x))
    I_result = target_abs**2
    I_result_mean = I_result.mean()

    for i in range(num_steps):
        print("{}/{}".format(i, num_steps), end='\r')
        DOE = far_field.ifft(z=z, shift=False)
        mask = np.angle(DOE.u)
        DOE.u = np.exp(1j * mask)
        field_z = DOE.fft(z=z, shift=False, matrix=True)
        I_z = np.abs(field_z)**2
        I_z = I_z * I_result_mean / I_z.mean()
        far_field.u = target_abs * np.exp(1j * np.angle(field_z))

        errors[i] = compute_error(I_result, I_z)

    if False:
        plt.figure()
        plt.imshow(I_result)
        plt.colorbar()
        plt.figure()
        plt.imshow(I_z)
        plt.colorbar()

    plt.plot(errors, 'k', lw=2)
    plt.title('errors')

    mask = np.fft.fftshift(mask)
    mask = (mask + np.pi) / (2 * np.pi)

    mask_final = Scalar_mask_XY(x, y, wavelength)
    mask_final.u = mask

    return mask_final, errors


def GS_Fresnel_algorithm_backup(source,
                                target,
                                z,
                                is_binary,
                                num_steps=5,
                                has_draw=False):

    x = target.x
    y = target.y
    wavelength = target.wavelength

    radius_x = (x.max() - x.min()) / 2
    radius_y = (y.max() - y.min()) / 2
    x_mean = (x.max() + x.min()) / 2
    y_mean = (y.max() + y.min()) / 2

    circle = Scalar_mask_XY(x, y, wavelength)

    # if has_mask:
    #     circle.circle(r0=(x_mean,y_mean), radius=(radius_x,radius_y))

    if source is None:
        source = Scalar_source_XY(x, y, wavelength)
        source.plane_wave()

    DOE = Scalar_mask_XY(x, y, wavelength)
    for i in range(num_steps):
        print("{}/{}".format(i, num_steps), end='\r')
        far_field = (source * DOE).RS(z=z, new_field=True)
        far_field.u = np.sqrt(np.abs(target.u)) * np.exp(
            1j * np.angle(far_field.u))
        DOE = far_field.RS(z=-z, new_field=True)
        mask = np.angle(DOE.u)

        # if has_mask:
        #     DOE.u=np.exp(1j*np.pi*mask)*circle.u

    mask = (mask + np.pi) / (2 * np.pi)
    print(mask.max(), mask.min())

    if is_binary:
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

    mask_final = Scalar_mask_XY(x, y, wavelength)
    mask_final.u = mask

    return mask_final


def verify_mask(x,
                y,
                wavelength,
                filename,
                z,
                has_mask,
                is_phase,
                is_binary,
                has_draw=False,
                has_axis=False,
                is_logarithm=True):
    """
    # TODO:
    """

    if isinstance(filename, str):
        name, _ = filename.split('.')
        DOE_new = Scalar_mask_XY(x, y, wavelength)
        DOE_new.image(filename)

    else:
        DOE_new = filename

    radius_x = (DOE_new.x.max() - DOE_new.x.min()) / 2
    radius_y = (DOE_new.y.max() - DOE_new.y.min()) / 2
    x_mean = (DOE_new.x.max() + DOE_new.x.min()) / 2
    y_mean = (DOE_new.y.max() + DOE_new.y.min()) / 2

    if is_phase:
        if is_binary:
            DOE_new.u = np.exp(1j * np.pi * DOE_new.u)
        else:
            DOE_new.u = np.exp(2j * np.pi * DOE_new.u)

    if has_mask:
        circle = Scalar_mask_XY(x, y, wavelength)
        circle.circle(r0=(x_mean, y_mean), radius=(radius_x, radius_y))
        DOE_new = DOE_new * circle

    if False:
        if is_phase:
            DOE_new.draw('phase')
            plt.title('phase')
        else:
            DOE_new.draw('intensity', colormap_kind='gray')
            plt.title('intensity')

    if z is None:
        result = DOE_new.fft(new_field=True, shift=True, remove0=True)

    else:
        result = DOE_new.RS(z=z, new_field=True)

    if is_phase:
        code = 'p'
    else:
        code = 'a'

    if has_draw:
        result.draw(logarithm=is_logarithm)
        plt.axis('off')
        if has_axis is True:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(name + code + '.png')
        else:
            intensity = np.abs(result.u)**2
            if is_logarithm:
                intensity = np.log(intensity + is_logarithm)
            plt.imsave(name + code + '.png',
                       intensity,
                       cmap='gist_heat',
                       dpi=300,
                       origin='lower')

    return DOE_new, result


def compute_error(I_result, I_target):
    """
    Computer the error as the average difference of the absolute value between of the intensity at target and the intensity at the result.

    Parameters:
        I_result (numpy.array): intensity produced by the algorithms
        I_target (numpy.array): intentisty at target
        is_balanced (bool): If True, perform the comparison using a proportion parameter to avoid different intensity levels


    Returns:
        error: Mean difference between result and target.

    """

    error = (np.abs(I_result - I_target)).mean()

    return error


"""
       if is_amplitude is True:
            mask=np.abs(DOE.u)
            if is_binary:
                is_0 = (mask<=mask.mean())
                mask[is_0]=0
                mask[~is_0]=1
            DOE.u=mask
            print(mask.mean())

        else:
            mask=np.angle(DOE.u)
            if is_binary:
                is_PI= (mask>np.pi/2) &  (mask<3*np.pi/2)
                mask[is_PI]=np.pi
                mask[~is_PI]=0
            DOE.u=np.exp(1j*mask)
"""


def make_binary(mask):
    """
    # TODO:
    """
    mask_new = Scalar_mask_XY(mask.x, mask.y, mask.wavelength)
    mask_new.u = deepcopy(mask.u)
    mask_new.u[mask.u > 0.5] = 1.
    mask_new.u[mask.u <= 0.5] = 0.
    return mask_new

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Name:        common.py
# Purpose:     Common functions to classes
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2017
# Copyright:   AOCG / UCM
# Licence:     GPL
# ----------------------------------------------------------------------
""" Common functions to classes """

import datetime

import hickle
import numpy as np
import scipy as sp


def several_propagations(iluminacion, masks, distances):
    '''performs RS propagation through several masks

    Parameters:
        iluminacion (Scalar_source_XY): illumination
        masks (list): list with several (Scalar_masks_XY)
        distances (list): list with seera distances


    Returns:
        Scalar_field_XY: u0 field at the last plane given by distances
        Scalar_field_XY: u1 field just at the plane of the last mask
    '''

    u0 = iluminacion

    for mask, distance in zip(masks, distances):
        u1 = u0 * mask
        u0 = u1.RS(z=distance)

    return u0, u1  # en el último plano y justo despues


def get_date():
    """gets current date and hour.

    Returns:
        (str): date in text
    """
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S")
    return date


def save_data_common(cls, filename='', method='', description=''):
    """Common save data function to be used in all the modules.

    Parameters:
        filename (str): filename
        method (str): saving method: savez, savez_compressed hickle, matlab, (h5py not yet)
        description (str): text to be stored in the dictionary to save.
    """

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S")

    cls.__dict__['date'] = date
    cls.__dict__['description'] = description

    if method == 'savez':
        np.savez(file=filename, dict=cls.__dict__)

    elif method == 'savez_compressed':
        np.savez_compressed(file=filename, dict=cls.__dict__)

    elif method == 'hickle':
        hickle.dump(cls.__dict__, filename, mode='w', compression='gzip')

    elif method == 'matlab':
        sp.io.savemat(filename, cls.__dict__)

    # elif method == 'h5py':
    #     with h5py.File(filename, 'w', libver='latest') as f:
    #         for k, v in cls.__dict__.items():
    #             f.create_dataset('dict/' + str(k), data=v, chunks=True)
    #         f.attrs['date'] = date


def load_data_common(cls, filename, method, verbose=False):
    """Common load data function to be used in all the modules.

    Parameters:
        filename (str): filename
        method (str): saving method: savez, savez_compressed hickle, matlab, (h5py not yet)
        verbose (bool): If True prints data
    """

    def print_data_dict(dict0):
        for k, v in dict0.items():
            print("{:12} = {}".format(k, v))
        print("\nnumber of data = {}".format(len(dict0['x'])))

    try:
        if method == 'hickle':
            dict0 = hickle.load(filename)

        elif method == 'savez' or 'savez_compressed':
            npzfile = np.load(filename)

            dict0 = npzfile['dict'].tolist()

        if verbose is True:
            print_data_dict(dict0)

        return dict0

    except IOError:
        print('could not open {}'.format(filename))
        return None

    # with h5py.File('file.h5', 'r', libver='latest') as f:
    #     x_read = f['dict']['X'][:]  # [:] syntax extracts numpy array into memory
    #     y_read = f['dict']['Y'][:]


def print_axis_info(cls, axis):
    """Prints info about axis

    Parameters:
        cls (class): class of the modulus.
        axis (): axis x, y, z... etc.
    """

    x0 = eval("cls.{}[0]".format(axis))
    x1 = eval("cls.{}[-1]".format(axis))
    length = x1 - x0
    Dx = eval("cls.{}[1]-cls.{}[0]".format(axis, axis))
    axis_info = dict(axis=axis, min=x0, max=x1, length=length, Dx=Dx)
    print("   axis={axis}: min={min}, max={max}, length={length}, Dx={Dx}".
          format(**axis_info))


def date_in_name(filename):
    """introduces a date in the filename.

    Parameters:
        filename (str): filename

    Returns:
        (str): filename with current date
    """
    divided = filename.split(".")
    extension = divided[-1]
    rest = divided[0:-1]
    initial_name = ".".join(rest)
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S_%f")
    filename_2 = "{}_{}.{}".format(initial_name, date, extension)
    return filename_2

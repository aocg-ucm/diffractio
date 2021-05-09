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

import numpy as np
from scipy.io import loadmat, savemat


def clear_all():
    """clear all variables"""
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]


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


def save_data_common(cls, filename, add_name='', description='', verbose=False):
    """Common save data function to be used in all the modules.
    The methods included are: npz, matlab


    Parameters:
        filename (str): filename
        add_name= (str): sufix to the name, if 'date' includes a date
        description (str): text to be stored in the dictionary to save.
        verbose (bool): If verbose prints filename.

    Returns:
        (str): filename. If False, file could not be saved.
    """

    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H_%M_%S")

    if add_name == 'date':
        add_name = "_" + date
    extension = filename.split('.')[-1]
    file = filename.split('.')[0]
    final_filename = file + add_name + '.' + extension

    if verbose:
        print(final_filename)

    cls.__dict__['date'] = date
    cls.__dict__['description'] = description

    if extension == 'npz':
        np.savez_compressed(file=final_filename, dict=cls.__dict__)

    elif extension == 'mat':
        savemat(final_filename, cls.__dict__)

    return final_filename


def load_data_common(cls, filename, verbose=False):
    """Common load data function to be used in all the modules.
        The methods included are: npz, matlab


    Parameters:
        cls (class): class X, XY, XZ, XYZ, etc..
        filename (str): filename
        verbose (bool): If True prints data
    """
    def print_data_dict(dict0):
        for k, v in dict0.items():
            print("{:12} = {}".format(k, v))
        print("\nnumber of data = {}".format(len(dict0['x'])))

    extension = filename.split('.')[-1]

    try:
        if extension in ('npy', 'npz'):
            npzfile = np.load(filename, allow_pickle=True)
            dict0 = npzfile['dict'].tolist()

        elif extension == 'mat':
            dict0 = loadmat(file_name=filename, mdict=cls.__dict__)

        else:
            print("extension not supported")

        if verbose is True:
            print(dict0.keys())

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

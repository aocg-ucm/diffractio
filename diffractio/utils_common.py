# !/usr/bin/env python3


# ----------------------------------------------------------------------
# Name:        utils_common.py
# Purpose:     Common utility functions for various operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2017
# Copyright:   AOCG / UCM
# Licence:     GPL
# ----------------------------------------------------------------------

""" Common functions to classes """
# flake8: noqa

import datetime
import multiprocessing

import numpy as np
import psutil
from scipy.io import loadmat, savemat
from scipy.ndimage import center_of_mass

from .config import Options_add



def check_none(*variables, raise_exception=False):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            for variable in variables:
                if getattr(self, variable) is None:
                    if raise_exception:
                        raise ValueError(f"{variable} is None")
                    else:
                        print(f"{variable} is None")
                    return  # Return immediately, do not execute the method
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def oversample(self, kind: str, factor_rate: int | tuple = (2,2)):
    """_summary_

    Args:
        factor_rate (tuple, optional): _description_. Defaults to (2,2).
    """


    if kind = 'scalar_x':

        self.x = np.linspace(self.x[0], self.x[-1], factor_rate[0]*len(self.x))
        self.u =  new_matrix.repeat(factor_rate[0])


    elif kind = 'scalar_xy':

        self.x = np.linspace(self.x[0], self.x[-1], factor_rate[0]*len(self.x))
        self.y = np.linspace(self.y[0], self.y[-1], factor_rate[0]*len(self.y))
        self.X, self.Y = np.meshgrid(self.x, self.y)

        new_matrix =  self.u.repeat(factor_rate[0],axis=0)
        self.u =  new_matrix.repeat(factor_rate[0],axis=1)

    elif kind = 'scalar_xyz':

        self.x = np.linspace(self.x[0], self.x[-1], factor_rate[0]*len(self.x))
        self.y = np.linspace(self.y[0], self.y[-1], factor_rate[0]*len(self.y))
        self.z = np.linspace(self.y[0], self.y[-1], factor_rate[0]*len(self.y))
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z)

        new_matrix =  self.u.repeat(factor_rate[0],axis=0)
        new_matrix =  new_matrix(factor_rate[1],axis=1)
        self.u =  new_matrix.repeat(factor_rate[2],axis=2)


        new_matrix =  self.n.repeat(factor_rate[0],axis=0)
        new_matrix =  new_matrix(factor_rate[1],axis=1)
        self.n =  new_matrix.repeat(factor_rate[2],axis=2)


    elif kind = 'scalar_xz':

        self.x = np.linspace(self.x[0], self.x[-1], factor_rate[0]*len(self.x))
        self.z = np.linspace(self.z[0], self.z[-1], factor_rate[0]*len(self.z))
        self.X, self.Y = np.meshgrid(self.x, self.y)

        new_matrix =  self.u.repeat(factor_rate[0],axis=0)
        self.u =  new_matrix.repeat(factor_rate[0],axis=1)

        new_matrix =  self.n.repeat(factor_rate[0],axis=0)
        self.n =  new_matrix.repeat(factor_rate[0],axis=1)


    return self


def add(self, other, kind: Options_add  = 'source'):
    """adds two fields. For example two light sources or two masks. The fields are added as complex numbers and then normalized so that the maximum amplitude is 1.
    
    Args:
        other (Vector_field_X): 2nd field to add
        kind (str): instruction how to add the fields: ['source', 'mask', 'phases', 'no_overlap', 'distances'].
            - 'source': adds the fields as they are
            - 'mask': adds the fields as complex numbers and then normalizes so that the maximum amplitude is 1.
            - 'phases': adds the phases and then normalizes so that the maximum amplitude is 1.
            - 'np_overlap': adds the fields as they are. If the sum of the amplitudes is greater than 1, an error is produced
            - 'distances': adds the fields as they are. If the fields overlap, the field with the smallest distance is kept.

    Returns:
        sum of the two fields.
    """

    
    from diffractio.scalar_sources_X import Scalar_source_X
    from diffractio.scalar_sources_XY import Scalar_source_XY
    from diffractio.scalar_masks_X import Scalar_mask_X
    from diffractio.scalar_masks_XY import Scalar_mask_XY
    

    if isinstance(self, Scalar_mask_XY):
        t = Scalar_mask_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_source_XY):
        t = Scalar_source_XY(self.x, self.y, self.wavelength)
    elif isinstance(self, Scalar_mask_X):
        t = Scalar_mask_X(self.x, self.wavelength)
    elif isinstance(self, Scalar_source_X):
        t = Scalar_source_X(self.x,  self.wavelength)


    if kind == 'source':
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t.u += o.u
        else:        
            t.u = self.u + other.u
    
    elif kind == 'mask':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        if isinstance(other, tuple):

            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u += o.u
                i_change = t1+t2>1
                t.u[i_change]=(np.exp(1j*f1[i_change])+np.exp(1j*f2[i_change])).astype(np.complex128)
                t.u[i_change]= t.u[i_change]/np.abs( t.u[i_change])
        else:
            t2 = np.abs(other.u)
            f2 = np.angle(other.u)

            t.u = self.u + other.u
            i_change = t1+t2>1
            t.u[i_change]=(np.exp(1j*f1[i_change])+np.exp(1j*f2[i_change])).astype(np.complex128)
            t.u[i_change]= t.u[i_change]/np.abs( t.u[i_change])
    
    elif kind == 'phases':
        t1 = np.abs(self.u)
        f1 = np.angle(self.u)
        
        if isinstance(other, tuple):
            t.u = self.u
            for o in other:
                t2 = np.abs(o.u)
                f2 = np.angle(o.u)
                t.u += o.u
                t.u = t1 * np.exp(1j * (f1 + f2))
        else:
            t2 = np.abs(other.u)
            f2 = np.angle(other.u)
        
            ts = t1 + t2
            ts[ts > 0] = 1.
            t.u = ts * np.exp(1j * (f1 + f2))

    elif kind == 'no_overlap':
        
        if isinstance(other, tuple):
            t.u = self.u
            for i, o in enumerate(other):
                i_pos1 = np.abs(t.u)>0
                i_pos2 = np.abs(o.u)>0
                print((i_pos1*i_pos2).sum())
                if (i_pos1 & i_pos2).any():
                    raise ValueError('The field {i} overlap with a previous one')
                t.u += o.u
        else:
            t1 = np.abs(self.u)
            t2 = np.abs(other.u)
            i_pos1 = t1>0
            i_pos2 = t2>0
            if (i_pos1 & i_pos2).any():
                raise ValueError('The two fields overlap')
            
    elif kind == 'distances':
        if isinstance(other, tuple): #todo: with simultaneous control of distances, not with for loop.
            pass
            
            """t.u = self.u

            for o in other:
                t.u = t.u + o.u
                com3 = center_of_mass(np.abs(t.u)>0)
                com4 = center_of_mass(np.abs(o.u)>0)
                
                t_max =  np.abs(t.u)>0
                o_max =  np.abs(o.u)>0
                overlap = t_max * o_max
                
                x3_center = t.x[int(com3[0])]
                x4_center = o.x[int(com4[0])]
                
                dist_t = np.abs(t.x-x3_center) * (np.abs(t.u)>0)
                dist_o = np.abs(o.x-x4_center) * (np.abs(o.u)>0)

                t.u = t.u+o.u
                i_menor = dist_t<dist_o
                i_mayor = dist_t>=dist_o

                t.u[i_menor*overlap] = t.u[i_menor*overlap]
                t.u[i_mayor*overlap] = o.u[i_mayor*overlap]"""
        else:  #only for 1D    
            print("not valid for 2D")
            t.u = self.u + other.u
            com3 = center_of_mass(np.abs(self.u)>0)
            com4 = center_of_mass(np.abs(other.u)>0)
            
            self_max =  np.abs(self.u)>0
            other_max =  np.abs(other.u)>0
            overlap = self_max * other_max
            
            x3_center = self.x[int(com3[0])]
            x4_center = other.x[int(com4[0])]
            
            dist_self = np.abs(self.x-x3_center) * (np.abs(self.u)>0)
            dist_other = np.abs(other.x-x4_center) * (np.abs(other.u)>0)

            t.u = self.u+other.u
            i_menor = dist_self<dist_other
            i_mayor = dist_self>=dist_other

            t.u[i_menor*overlap] = self.u[i_menor*overlap]
            t.u[i_mayor*overlap] = other.u[i_mayor*overlap]
        
    return t



def computer_parameters(verbose: bool = False) -> tuple[int, float, float, float]:
    """Determine several computer Args:
        - number of processors
        - available memory
        - total memory
        - max frequency

    Args:
        verbose (bool): If True prints data

    Returns:
        num_max_processors (int): number of processors
        info_memory (int) : Gbytes
        memory_available (int): % available memory
        freq_max (int): Maximum frequency (GHz)
    """

    freq_max = psutil.cpu_freq()
    info_memory = psutil.virtual_memory()[0] / 1024**3
    memory_available = psutil.virtual_memory(
    ).available * 100 / psutil.virtual_memory().total

    num_max_processors = multiprocessing.cpu_count()

    if verbose:
        print("number of processors: {}".format(num_max_processors))
        print("total memory        : {:1.1f} Gb".format(info_memory))
        print("available memory    : {:1.0f} %".format(memory_available))
        print("max frequency       : {:1.0f} GHz".format(freq_max[2]))

    return num_max_processors, info_memory, memory_available, freq_max[2]


def clear_all():
    """clear all variables"""
    all = [var for var in globals() if var[0] != "_"]
    for var in all:
        del globals()[var]


def several_propagations(source, masks, distances: list[float]):
    '''performs RS propagation through several masks

    Args:
        source (Scalar_source_XY): illumination
        masks (list): list with several (Scalar_masks_XY)
        distances (list): list with seera distances


    Returns:
        Scalar_field_XY: u0 field at the last plane given by distances
        Scalar_field_XY: u1 field just at the plane of the last mask
    '''

    u0 = source

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


def save_data_common(cls,
                     filename: str,
                     add_name: str = '',
                     description: str = '',
                     verbose: bool = False) -> str:
    """Common save data function to be used in all the modules.
    The methods included are: npz, matlab

    Args:
        filename(str): filename
        add_name = (str): sufix to the name, if 'date' includes a date
        description(str): text to be stored in the dictionary to save.
        verbose(bool): If verbose prints filename.

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


def load_data_common(cls, filename: str, verbose: bool = False):
    """Common load data function to be used in all the modules.
        The methods included are: npz, matlab

    Args:
        cls(class): class X, XY, XZ, XYZ, etc..
        filename(str): filename
        verbose(bool): If True prints data
    """

    def print_data_dict(dict0: dict):
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

        if dict0 is not None:
            if isinstance(dict0, dict):
                cls.__dict__ = dict0
            else:
                raise Exception('no dictionary in load_data')

        return dict0

    except IOError:
        print('could not open {}'.format(filename))
        return None

    # with h5py.File('file.h5', 'r', libver='latest') as f:
    #     x_read = f['dict']['X'][:]  # [:] syntax extracts numpy array into memory
    #     y_read = f['dict']['Y'][:]


def print_axis_info(cls, axis: str):
    """Prints info about axis

    Args:
        cls(class): class of the modulus.
        axis(): axis x, y, z... etc.
    """

    x0 = eval("cls.{}[0]".format(axis))
    x1 = eval("cls.{}[-1]".format(axis))
    length = x1 - x0
    Dx = eval("cls.{}[1]-cls.{}[0]".format(axis, axis))
    axis_info = dict(axis=axis, min=x0, max=x1, length=length, Dx=Dx)
    print("   axis={axis}: min={min}, max={max}, length={length}, Dx={Dx}".
          format(**axis_info))


def date_in_name(filename: str) -> str:
    """introduces a date in the filename.

    Args:
        filename(str): filename

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

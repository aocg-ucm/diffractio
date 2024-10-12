#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Name:        utils_tests.py
# Purpose:     Utility functions for testing operations
#
# Author:      Luis Miguel Sanchez Brea
#
# Created:     2024
# Licence:     GPLv3
# ----------------------------------------------------------------------


# flake8: noqa

""" Common functions to classes """

import datetime
import multiprocessing
import sys
import time

from .utils_typing import NDArrayFloat

from .__init__ import mm, no_date, np, plt, um
from .scalar_masks_XY import Scalar_mask_XY

max_num_cores = multiprocessing.cpu_count()
min_num_pixels = 8
max_num_pixels = 11

NUM_CORES = np.array(list(range(1, max_num_cores + 1)))
n = np.array(list(range(min_num_pixels, max_num_pixels)))
NUM_PIXELS = 2**n
NUM_PROCESSES = max_num_cores  # 8

if no_date is True:
    date = '0'
else:
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d_%H")

path_base = "tests_results"
path_class = "utils_math"

newpath = "{}_{}/{}/".format(path_base, date, path_class)


def _test_slit_RS_XY(num_pixels: int):
    """_test_slit_RS_XY

    Args:
        num_pixels (int): num_pixels

    Returns:
        u: field
    """
    length = 512*um
    x = np.linspace(-length/2, length/2, num_pixels)
    y = np.linspace(-length/2, length/2, num_pixels)
    wavelength = 0.6328*um

    u1 = Scalar_mask_XY(x, y, wavelength)
    u1.slit(x0=0, size=25*um)
    u1.RS(z=1*mm, new_field=False, verbose=False)
    return u1


def run_benchmark(num_pixels: int):
    """_summary_

    Args:
        num_pixels (int): _description_
    """

    _test_slit_RS_XY(num_pixels=num_pixels)

    def test_pixels(self):
        func_name = sys._getframe().f_code.co_name
        # class_name = self.__class__.__name__

        num_pixeles = 256
        time_array = benchmark_processors_n_pixels(num_pixeles) / NUM_PROCESSES

        plt.figure()
        plt.plot(NUM_CORES, time_array)
        plt.xlabel('num_cores')
        plt.ylabel('time_array')
        plt.title('num_pixels: {}, num_processes: {}'.format(
            str(num_pixeles), str(NUM_PROCESSES)))
        save_figure_test(newpath, func_name, '_time_numpixels')

        plt.figure()
        plt.plot(NUM_CORES, time_array[0] / time_array)
        plt.title('num_pixels: {}, num_processes: {}'.format(
            str(num_pixeles), str(NUM_PROCESSES)))
        plt.xlabel('num_cores')
        plt.ylabel('aceleracion')
        save_figure_test(newpath, func_name, '_acc')


def comparison(proposal: NDArrayFloat, solution: NDArrayFloat, maximum_diff: float):
    """This functions is mainly for testing. It compares compares proposal to solution.

    Args:
        proposal (numpy.matrix): proposal of result.
        solution (numpy.matrix): results of the test.
        maximum_diff (float): maximum difference allowed.

    Returns:
        (bool): True if comparison is possitive, else False.
    """

    comparison1 = np.linalg.norm(proposal - solution) < maximum_diff

    return comparison1


def save_figure_test(newpath: str, func_name: str, add_name: str = ''):
    """_summary_

    Args:
        newpath (str): _description_
        func_name (str): _description_
        add_name (str, optional): _description_. Defaults to ''.
    """
    title = '{}{}'.format(func_name, add_name)
    plt.suptitle(title)
    filename = '{}{}{}.{}'.format(newpath, func_name, add_name, 'png')
    plt.savefig(filename)
    plt.close('all')


def ejecute_multiprocessing(num_cores: int, n_pixels: int):
    """_summary_

    Args:
        num_cores (int): _description_
        n_pixels (int): _description_
    """
    num_pixeles = n_pixels * np.ones(NUM_PROCESSES)
    if num_cores == 1:
        [run_benchmark(i_pixels) for i_pixels in num_pixeles]
    else:
        pool = multiprocessing.Pool(processes=num_cores)
        pool.map(run_benchmark, num_pixeles)
        pool.close()
        pool.join()


def benchmark_num_pixels(function, n_max: int = 10):
    """This function is for benchmarking using an increasing number of pixels 2**n.

    Args:
        function (function): Functions that has as argument the number of pixels 2**n.
    """

    n = np.array(range(6, n_max + 1))
    NUM_PIXELS = 2**n
    time_array = np.zeros_like(NUM_PIXELS, dtype='double')

    for n_pixels, i in zip(NUM_PIXELS, range(len(NUM_PIXELS))):
        t1 = time.time_ns()()
        function(num_data=n_pixels)
        t2 = time.time_ns()
        time_array[i] = (t2 - t1) * 1000
        print(n[i], n_pixels, t1, t2, time_array[i])

    plt.figure()
    plt.plot(n, time_array, 'ko', ms=12)
    plt.figure()
    plt.plot(NUM_PIXELS, time_array / NUM_PIXELS, 'ko', ms=12)


def benchmark_processors_n_pixels(n_pixels: int):
    """_summary_

    Args:
        n_pixels (int): _description_

    Returns:
        _type_: _description_
    """

    time_array = np.zeros_like(NUM_CORES, dtype='float')
    for i, core in enumerate(NUM_CORES):
        t1 = time.time()
        ejecute_multiprocessing(num_cores=core, n_pixels=n_pixels)
        t2 = time.time()
        time_array[i] = t2 - t1
    return time_array


def save_data_test(cls, newpath: str, func_name: str, add_name: str = ''):
    """_summary_

    Args:
        newpath (_type_): _description_
        func_name (_type_): _description_
        add_name (str, optional): _description_. Defaults to ''.
    """

    filename = '{}{}{}.{}'.format(newpath, func_name, add_name, 'npz')
    print(filename)
    np.savez_compressed(file=filename, dict=cls.__dict__)

# !/usr/bin/env python3


# https://docs.python.org/3/library/typing.html
# https://numpy.org/devdocs/reference/typing.html


import numpy as np
from typing import Any, List
import numpy.typing as npt


NDArray = npt.NDArray
NDArrayInt = npt.NDArray[np.integer]
NDArrayFloat = npt.NDArray[np.float]
NDArrayComplex = npt.NDArray[np.complexfloat]
float = np.float
integer = np.integer


# Para 3.12 o posterior, es decir, es muy nuevo
# type point2D = list[float, float] | NDArray

# some temporal helps
"""
type Vector = list[float]
type ConnectionOptions = dict[str, str]
type Address = tuple[str, int]
type Server = tuple[Address, ConnectionOptions]

np.array(x**2 for x in range(10))  # type: ignoref

from typing import NewType

UserId = NewType('UserId', int)
some_id = UserId(524313)
list[int, str]
tuple[int] = (1, 2, 3)
tuple[int, ...] = (1, 2)

clases
def make_new_user(user_class: type[User]) -> User:
 
from typing import NoReturn

def stop() -> NoReturn:
    raise RuntimeError('no way')
"""

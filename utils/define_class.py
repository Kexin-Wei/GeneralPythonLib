from pathlib import Path
import numpy as np
from typing import Union, Sequence, Tuple
from enum import Enum


class DatetimeChangingType(Enum):
    pure = "pure"
    append = "append"


class AverageMethodType(Enum):
    median = "median"
    average = "average"


class LsOptionType(Enum):
    dir = "dir"
    file = "file"


class TwoDConnectionType(Enum):
    four = "4-point-connected"
    eight = "8-point-connected"


class NeighbourPackingType(Enum):
    hexagonal = "hexagonal"
    line = "line"


class ImageSourceType(Enum):
    probe = "probe"
    image = "image"


INT_OR_FLOAT = Union[int, float]
STR_OR_LIST = Union[str, Sequence]

PATH_OR_LIST = Union[Path, Sequence[Path]]
STR_OR_PATH = Union[str, Path]

LIST_OR_NUMPY = Union[Sequence, np.ndarray]
INT_OR_NUMPY = Tuple[np.ndarray, int]

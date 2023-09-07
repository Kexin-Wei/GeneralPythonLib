from pathlib import Path
import numpy as np
from typing import Union, Sequence, Tuple
from enum import Enum

# PURE_OR_APPEND = Literal["pure", "append"]
# AC_METHOD = Literal["median", "average"]
# DIR_OR_FILE = Literal["dir", "file"]
# FOUR_OR_EIGHT = Literal["4-point-connected", "8-point-connected"]
# NEIGHBOUR_PACKING_TYPE = Literal["hexagonal", "line"]
# PROBE_OR_IMAGE = Literal["probe", "image"]


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


class JointType(Enum):
    revolute = "revolute"
    prismatic = "prismatic"


class Dimension(Enum):
    two = 2
    three = 3


INT_OR_FLOAT = Union[int, float]
STR_OR_LIST = Union[str, Sequence]

PATH_OR_LIST = Union[Path, Sequence[Path]]
STR_OR_PATH = Union[str, Path]

LIST_OR_NUMPY = Union[Sequence, np.ndarray]
INT_OR_NUMPY = Tuple[np.ndarray, int]

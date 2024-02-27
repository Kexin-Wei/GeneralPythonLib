import datetime
from pathlib import Path

from .define_class import DatetimeChangingType


def datetimeChangingFolder(parentPath: Path,
                           mode: DatetimeChangingType = DatetimeChangingType.pure) -> Path:
    if mode == DatetimeChangingType.pure:
        return parentPath.joinpath(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if mode == DatetimeChangingType.pure:
        return parentPath.joinpath(f"{parentPath.name}_"f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    print(f"Mode {mode} not supported")
    return Path()

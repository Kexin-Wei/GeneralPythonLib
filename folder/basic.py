"""created by kx 20221104
Several folder Managers based on folderMeg, including:
    - pgmFolderMg: manage the pgm files inside the folder
    - parentFolderMg: manage the child directories inside the folder
"""

import natsort
import warnings
import numpy as np
from pathlib import Path
from typing import Sequence, Optional, List

from ..utils.define_class import (
    STR_OR_PATH,
    STR_OR_LIST,
    PATH_OR_LIST,
    LsOptionType,
)


class FolderMgBase:
    """
    A Base Class to manage files and folders inside a parent folder, functions include:
        - set path
        - get a list of file names and another list of directories name
        - tags function:
            - add tag to the pgmFolder, eg, "f3.5","lowAttention"
            - check whether a tag is inside pgmFolder
            - list all the tags
    """

    def __init__(self):
        self.full_path: Path = None
        self.parentFolder = None
        self.folderName = None
        self.tags = None
        self.dirs: Sequence[Path] = None
        self.files: Sequence[Path] = None

    @staticmethod
    def _str_to_list(anyStr: STR_OR_LIST) -> list:
        if isinstance(anyStr, str):
            return [anyStr]
        if isinstance(anyStr, list):
            return anyStr
        print("Input is not a str or list.")
        return []

    @staticmethod
    def _path_to_list(anyPath: PATH_OR_LIST) -> list:
        if isinstance(anyPath, Path):
            return [anyPath]
        if isinstance(anyPath, list):
            return anyPath
        print("Input is not a Path or list.")
        return []

    def _ge_files_dirs(self):
        if self.full_path is not None:
            self.dirs = [p for p in self.full_path.iterdir() if p.is_dir()]
            self.files = [f for f in self.full_path.iterdir() if f.is_file()]
            self.dirs = natsort.natsorted(self.dirs)
            self.files = natsort.natsorted(self.files)

    def _get_file_path_by_extension(self, extension: str) -> List[Path]:
        # put extension in a pure string without . and *, e.g. python file input "py"
        try:
            sorted_files = natsort.natsorted(self.full_path.glob(f"*.{extension}"))
        except Exception as e:
            warnings.warn(f"Failed to get files with extension {extension}, error: {e}.")
            sorted_files = []
        return sorted_files

    def _get_file_path_by_extension_list(self, extensions: list) -> List[Path]:
        files = []
        for e in extensions:
            files.extend(self._get_file_path_by_extension(e))
        return files

    def get_random_file(self, printOut: bool = True) -> Path:
        if self.files is not None and len(self.files) != 0:
            randomIdx = np.random.randint(low=0, high=len(self.files))
            if printOut:
                print(
                    f"Get File with idx: {randomIdx}, name: {self.files[randomIdx].name}, in folder: {self.folderName}"
                )
            return self.files[randomIdx]
        print(f"{self.folderName} contains NO files\n")
        return Path()

    @property
    def nFile(self) -> Optional[int]:
        if self.files is not None:
            return len(self.files)

    @property
    def nDirs(self) -> Optional[int]:
        if self.dirs is not None:
            return len(self.dirs)


class FolderMg(FolderMgBase):
    """
    functions:
        1. sort files by types
        2. ls files and dirs
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__()
        self.full_path = Path(folderFullPath)
        self.parentFolder = self.full_path.parent
        self.folderName = self.full_path.name
        self._ge_files_dirs()

    def ls(self, lsOption: Optional[LsOptionType] = None) -> None:
        if lsOption == LsOptionType.dir or lsOption is None:
            if self.dirs is None or len(self.dirs) == 0:
                print(f"\nCurrent Folder '{self.folderName}' contains NO folders\n")
            else:
                print(
                    f"\nCurrent Folder '{self.folderName}' contains {len(self.dirs)} folders, which are:"
                )
                for d in self.dirs[:5]:
                    print(f"  - {d.name}")
                if len(self.dirs) > 5:
                    print(f"  - ...")
        if lsOption == LsOptionType.file or lsOption is None:
            if self.files is None or len(self.files) == 0:
                print(f"\nCurrent Folder '{self.folderName}' contains NO files\n")
            else:
                print(
                    f"\nCurrent Folder '{self.folderName}' contains {len(self.files)} files, which are:"
                )
                for f in self.files[:5]:
                    print(f"  - {f.name}")
                if len(self.files) > 5:
                    print(f"  - ...")

    def get_file_group_by_extension(self) -> dict:
        fileGroup = {}
        if self.files is not None:
            for f in self.files:
                extension = f.suffix.lower()
                if extension in fileGroup:
                    fileGroup[extension].append(f)
                else:
                    fileGroup[extension] = [f]
        return fileGroup


class FolderTagMg(FolderMgBase):
    """
    add tags to list of folders manually
    """

    def __init__(
        self, fullPath: STR_OR_PATH = Path(), tags: Optional[STR_OR_LIST] = None
    ):
        if tags is None:
            tags = []
        super().__init__()
        self.full_path = Path(fullPath)
        self.parentFolder = self.full_path.parent
        self.folderName = self.full_path.name
        self._ge_files_dirs()
        self.tags = set(self._str_to_list(tags))

    def add_tags(self, tags: STR_OR_LIST):
        tags = self._str_to_list(tags)
        for t in tags:
            self.tags.add(t)

    def contains_tag(self, tag: str) -> bool:
        return tag in self.tags

    def ls_tags(self):
        print(f"\nCurrent Folder '{self.folderName}' contains tags:")
        for t in self.tags:
            print(f"  - {t}")


class URDFFolderMg(FolderMg):
    """
    Find all the urdf files for each robot
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__(folderFullPath)
        self.urdfs = {}

    def print_URDF_files(self):
        for k, v in self.urdfs.items():
            if isinstance(v, Path):
                print(f"\nIn folder {k}, there is one urdf file:")
                print(f"  - {v.name}")
            else:
                print(f"\nIn folder {k}, there are {len(v)} urdf files:")
                for f in v:
                    print(f"  - {f.name}")

    def get_URDF_from_all_dir(self):
        if self.nDirs:
            for d in self.dirs:
                urdfs = self.get_URDF(d)
                if len(urdfs):
                    if len(urdfs) > 1:
                        self.urdfs[d.name] = urdfs
                    else:
                        self.urdfs[d.name] = urdfs[0]

    @staticmethod
    def get_URDF(dirPath):
        urdfs = []
        dirMg = FolderMg(dirPath)
        if dirMg.nFile:
            for f in dirMg.files:
                if "urdf" in f.suffix.lower():
                    urdfs.append(f)
        return urdfs

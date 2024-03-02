import natsort
import warnings
from pathlib import Path
from typing import Dict, Optional

from ..utils.define_class import STR_OR_PATH, STR_OR_LIST, PATH_OR_LIST
from ..ultrasound.pgm import PGMFile
from .basic import FolderMgBase, FolderTagMg


class PgmFolder(FolderTagMg):
    """
    A Child Class of folderMgBase to manage the pgm files inside folder, functions include:
        - all parent functions
        - list all the pgm files
        - read pgm files inside the folder
        - read one pgm file in the list of file names, eg, readNextFile()

    """

    def __init__(self, fullPath: STR_OR_PATH, tags: Optional[STR_OR_LIST] = None):
        if tags is None:
            tags = []
        super().__init__(fullPath, tags)
        self._getPGMFiles()
        self.currentFileIndex = 0
        self.currentPgm = None

    def _getPGMFiles(self):
        return self._get_file_path_by_extension(".pgm")

    def ls(self):
        print(
            f"\nCurrent Folder '{self.folderName}' contains {len(self.files)} pgm files, which are:"
        )
        nListed = 0
        for f in self.files:
            print(f"  - {f.name}")
            nListed += 1
            if nListed > 20:
                print("  - ...")
                break

    def readCurrentFile(self, printOut: bool = True) -> PGMFile:
        if self.currentFileIndex == 0 and printOut:
            print(f"\nStart reading folder {self.full_path}")
        try:
            self.currentPgm = PGMFile(self.files[self.currentFileIndex], printOut=False)
        except IndexError:
            warnings.warn(f"Index {self.currentFileIndex} out of range.")
            return None
        if printOut:
            print(
                f"- File idx: {self.currentFileIndex}, name:{self.currentPgm.fileName}, in folder: {self.folderName}"
            )
        return self.currentPgm

    def readNextPgm(self):
        self.currentFileIndex += 1
        return self.readCurrentFile()

    def readRandomPgm(self, printOut: bool = True) -> PGMFile:
        return PGMFile(self.get_random_file(printOut=printOut), printOut=False)

    def saveBModes(
        self,
        imageRootFolderPath: Optional[STR_OR_PATH] = None,
        upperDisplayRangeDb: Optional[int] = None,
        lowerDisplayRangeDb: Optional[int] = None,
        replace: bool = False,
    ):
        if imageRootFolderPath is None:
            imageRootFolderPath = self.full_path.parent
        imageRootFolderPath = Path(imageRootFolderPath)
        imageFolderPath = imageRootFolderPath.joinpath(f"{self.folderName}_b-mode")
        for f in self.files:
            pgmFile = PGMFile(f, printOut=False)
            pgmFile.saveBMode(
                imageFolderPath,
                upperDisplayRangeDb=upperDisplayRangeDb,
                lowerDisplayRangeDb=lowerDisplayRangeDb,
                replace=replace,
            )


class ParentFolderTagMg(FolderTagMg):
    """
    A Child Class of folderMgBase to manage all the pgm Folder, functions include:
        - all parent functions
        - list all the dirs
        - return any folder as pgmFolderMg in the list given index of the list
    """

    def __init__(self, fullPath: STR_OR_PATH, tags: Optional[STR_OR_LIST] = None):
        if tags is None:
            tags = []
        super().__init__(fullPath, tags)

    def createNewFolderList(self, folders: STR_OR_LIST):
        # remove all the folders, and refill with new ones
        folders = self._str_to_list(folders)
        self.dirs = []
        for fd in folders:
            fdPath = self.full_path.joinpath(fd)
            if fdPath.exists():
                self.dirs.append(fdPath)
            else:
                print(f"Folder {fd} doesn't exist.")

    def ls(self):
        print(
            f"\nCurrent Folder '{self.folderName}' contains {len(self.dirs)}, which are:"
        )
        for d in self.dirs:
            print(f"  - {d.name}")

    def readPgmFolder(self, idx: int) -> PgmFolder:
        return PgmFolder(self.dirs[idx])


class PgmFolderTagMg(FolderMgBase):
    """
    A Child Class of folderMgBase to manage folders by tags:
        - all parent functions
        - add list of Path() or a single one with tags to group them
        - list all the included folders with their tags
        - return a list of folder given searched tags
    """

    def __init__(self, folders: Optional[PATH_OR_LIST] = None):
        if folders is None:
            folders = []
        super().__init__()
        self.folderList: Dict[
            Path, PgmFolder
        ] = dict()  # key = path, value = store the folderMgBase type of folders
        self.tagGroup: Dict[str, list[PgmFolder]] = dict()
        folders = self._path_to_list(folders)
        for fd in folders:
            self.folderList[fd] = PgmFolder(fd)

    def addTagsByFolderName(self, tags: STR_OR_LIST) -> None:
        """
        only for the folders in the self.folderList
        """
        tags = self._str_to_list(tags)
        for t in tags:
            if t not in self.tagGroup.keys():
                self.tagGroup[t] = []  # create new list for new tag.value
            for pgmFmg in self.folderList.values():
                if (
                    f"_{t}" in pgmFmg.folderName
                ):  # !!!HARDCODE:avoid find 7.5mhz using 5mhz
                    pgmFmg.add_tags(t)
                    self.tagGroup[t].append(pgmFmg)

    def addGroup(self, folders: PATH_OR_LIST, tags: STR_OR_LIST) -> None:
        tags = self._str_to_list(tags)
        folders = self._path_to_list(folders)

        for t in tags:
            if t not in self.tagGroup.keys():
                self.tagGroup[t] = []  # create new list for new tag.value

        for fd in folders:
            if fd not in self.folderList.keys():
                self.folderList[fd] = PgmFolder(fd)  # only create once
            self.folderList[fd].add_tags(tags)
            for t in tags:
                self.tagGroup[t].append(PgmFolder(fd))

    def findByTags(self, tags: STR_OR_LIST) -> list[PgmFolder]:
        tags = self._str_to_list(tags)
        if len(tags) == 1:
            if tags[0] in self.tagGroup.keys():
                return self.tagGroup[tags[0]]
            else:
                print(f"No such tag '{tags}' in the list of tags")
                return []

        resultGroup = set(self.tagGroup[tags[0]])
        for ithTag in range(len(tags) - 1):
            if not resultGroup:
                print(f"No common folder found.")
                return []
            resultGroup = resultGroup.intersection(self.tagGroup[tags[ithTag + 1]])
        return list(resultGroup)

    def lsByTag(self, tag: str) -> None:
        if tag in self.tagGroup.keys():
            print(f'\nTag "{tag}" contains folders')
            for fp in self.tagGroup[tag]:
                print(f"  - {fp.full_path}")
        else:
            print(f"No such tag '{tag}' in the list of tags")

    def ls(self) -> None:
        print(f"\nFolder Manager contains folders as following:")
        for fmg in self.folderList.values():
            print(f"- {fmg.folderName}", end="")
            if len(fmg.tags) != 0:
                print(", \ttags: ", end="")
                for t in natsort.natsorted(fmg.tags):
                    print(f"{t} ", end="")
                print()

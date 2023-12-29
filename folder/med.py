import SimpleITK as sitk
from pathlib import Path
from typing import Sequence, Optional, List, Union
from .basic import FolderMg
from ..utils.define_class import STR_OR_PATH
from ..med_image.medical_image import VolumeImage


class BaseMedicalImageFolderMg(FolderMg):
    """
    return images path given different image formats, currently supported
        - Meta Image: *.mha, *.mhd
        - Nifti Image: *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        - Nrrd Image: *.nrrd, *.nhdr
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__(folderFullPath)

    def getNrrdImagePath(self) -> List[Path]:
        # *.nrrd, *.nhdr
        return self._getFilePathByExtensionList(["nrrd", "nhdr"])

    def getMetaImagePath(self) -> List[Path]:
        # *.mha, *.mhd
        return self._getFilePathByExtensionList(["mha", "mhd"])

    def getNiftiImagePath(self) -> List[Path]:
        # *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        return self._getFilePathByExtensionList(
            ["nia", "nii", "nii.gz", "hdr", "img", "img.gz"]
        )


FOLDERMG_OR_PATH_OR_STR = Union[FolderMg, Path, str]


class T2FolderMg(FolderMg):
    """
    Find certain file in a net-structure folder, which has multiple folders that contain their own folders inside them
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__(folderFullPath)
        self.t2List = []

    def getT2(self):
        self.t2List.extend(self.searchT2inCurrentFolder())
        if self.nDirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = T2FolderMg(d)
                cMg.getT2()
                self.t2List.extend(cMg.t2List)

    def searchT2inCurrentFolder(self):
        if self.nFile:
            t2List = []
            for f in self.files:
                if "t2" in f.name.lower() and ("mha" in f.suffix or "nrrd" in f.suffix):
                    if "_cor" not in f.name.lower() and "_sag" not in f.name.lower():
                        t2List.append(f)
                        print(f)
            return t2List
        return []


class MedicalImageFolderMg(BaseMedicalImageFolderMg):
    """Add more dicom handling functions to BaseMedicalImageFolderMg

    Args:
        BaseMedicalImageFolderMg: return images path given different med_image formats, currently supported
            - Meta Image: *.mha, *.mhd
            - Nifti Image: *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
            - Nrrd Image: *.nrrd, *.nhdr

    """

    def __init__(self, folderFullPath: STR_OR_PATH):
        super().__init__(folderFullPath)
        self.dicomSeriesFolder: Optional[Sequence[Path]] = None
        self.dicomSeries: Optional[Sequence[VolumeImage]] = None
        self.findDicomSeriesFolder()
        self.readAllDicomSeries()

    def readAllDicomSeries(self) -> Sequence[VolumeImage]:
        """Read all dicom series from a folder

        Returns:
            Sequence[VolumeImage]: a list of dicom series
        """
        if self.dicomSeriesFolder is not None:
            print("No dicom series found, please read dicom series first")
            return []
        print("read all dicom series")
        for dicomFolder in self.dicomSeriesFolder:
            self.dicomSeries.append(self.readDicomSeries(dicomFolder))
            print(f"- {dicomFolder.name} read")
        return [self.readDicomSeries(folder) for folder in self.dicomSeriesFolder]

    def readDicomSeries(self, folderPath: STR_OR_PATH) -> VolumeImage:
        """Read dicom series from a folder

        Args:
            folderPath (STR_OR_PATH): folder path

        Returns:
            Sequence[Path]: a list of dicom files
        """
        if not self._isADicomSeries(folderPath):
            return []
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folderPath))
        series_file_name = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(folderPath), series_IDs[0]
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_name)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        img = series_reader.Execute()
        volumeImg = VolumeImage(img)
        return volumeImg

    def findDicomSeriesFolder(self):
        if self.dicomSeriesFolder is None:
            for folder in self.dirs:
                if self._isADicomSeries(folder):
                    print(f"- {folder.name} is a dicom series")
                    self.dicomSeriesFolder.append(folder)
                print(f"- {folder.name} is not a dicom series")
        print(
            f"found {len(self.dicomSeriesFolder)} dicom series already, skip this time"
        )

    def deleteDicomSeries(self):
        if self.dicomSeriesFolder is not None:
            self.dicomSeriesFolder = None
            print("delete dicom series")

    def _isADicomSeries(self, folderPath: STR_OR_PATH) -> bool:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folderPath))
        if not series_IDs:
            print(
                f"ERROR: given directory {folderPath} does not contain a DICOM series."
            )
            return False
        return True

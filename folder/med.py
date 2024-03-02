import SimpleITK as sitk
from pathlib import Path
from typing import List, Union
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

    def __init__(self, folder_full_path: STR_OR_PATH = Path()):
        super().__init__(folder_full_path)

    def get_nrrd_image_path(self) -> List[Path]:
        # *.nrrd, *.nhdr
        return self._get_file_path_by_extension_list(["nrrd", "nhdr"])

    def getMetaImagePath(self) -> List[Path]:
        # *.mha, *.mhd
        return self._get_file_path_by_extension_list(["mha", "mhd"])

    def getNiftiImagePath(self) -> List[Path]:
        # *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        return self._get_file_path_by_extension_list(
            ["nia", "nii", "nii.gz", "hdr", "img", "img.gz"]
        )


FOLDER_MG_OR_PATH_OR_STR = Union[FolderMg, Path, str]


class MedicalFolderMg(FolderMg):
    """
    Find certain file in a net-structure folder,
    which has multiple folders that contain their own folders inside them
    """

    def __init__(self, folder_full_path: STR_OR_PATH = Path()):
        super().__init__(folder_full_path)
        self.t2List = []
        self.dwiList = []
        self.adcList = []

    def search_T2_in_current_folder(self):
        if self.n_files:
            t2List = []
            for f in self.files:
                if "t2" in f.name.lower() and (
                    "mha" in f.suffix or "nrrd" in f.suffix
                ):
                    if (
                        "_cor" not in f.name.lower()
                        and "_sag" not in f.name.lower()
                    ):
                        t2List.append(f)
                        print(f)
            return t2List
        return []

    def search_dwi_in_current_folder(self):
        if self.n_files:
            dwiList = []
            for f in self.files:
                if "dwi" in f.name.lower() and (
                    "mha" in f.suffix or "nrrd" in f.suffix
                ):
                    dwiList.append(f)
                    print(f)
            return dwiList
        return []

    def search_adc_in_current_folder(self):
        if self.n_files:
            adcList = []
            for f in self.files:
                if "adc" in f.name.lower() and (
                    "mha" in f.suffix or "nrrd" in f.suffix
                ):
                    adcList.append(f)
                    print(f)
            return adcList
        return []

    def get_T2(self):
        self.t2List.extend(self.search_T2_in_current_folder())
        if self.n_dirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_T2()
                self.t2List.extend(cMg.t2List)

    def get_DWI(self):
        self.dwiList.extend(self.search_dwi_in_current_folder())
        if self.n_dirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_DWI()
                self.dwiList.extend(cMg.dwiList)

    def get_ADC(self):
        self.adcList.extend(self.search_adc_in_current_folder())
        if self.n_dirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_ADC()
                self.adcList.extend(cMg.adcList)


class DicomImageFolderMg(BaseMedicalImageFolderMg):
    """Add more dicom handling functions to BaseMedicalImageFolderMg:
    return images path given different med_image formats, currently supported
        - Meta Image: *.mha, *.mhd
        - Nifti Image: *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        - Nrrd Image: *.nrrd, *.nhdr

    """

    def __init__(self, folder_full_path: STR_OR_PATH):
        super().__init__(folder_full_path)
        self.dicomSeriesFolder: list[Path] = None
        self.dicomSeries: list[VolumeImage] = None
        self.find_dicom_series_folder()
        self.read_all_dicom_series()

    def read_all_dicom_series(self) -> list[VolumeImage]:
        """Read all dicom series from a folder

        Returns:
            Sequence[VolumeImage]: a list of dicom series
        """
        if self.dicomSeriesFolder is None:
            print("No dicom series found, please read dicom series first")
            return []
        print("read all dicom series")
        for dicomFolder in self.dicomSeriesFolder:
            self.dicomSeries.append(self.read_dicom_series(dicomFolder))
            print(f"- {dicomFolder.name} read")
        return [
            self.read_dicom_series(folder) for folder in self.dicomSeriesFolder
        ]

    def read_dicom_series(self, folder_path: STR_OR_PATH) -> VolumeImage | None:
        """Read dicom series from a folder

        Args:
            folder_path (STR_OR_PATH): folder path

        Returns:
            Sequence[Path]: a list of dicom files
        """
        if not self._is_a_dicom_series(folder_path):
            return None
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder_path))
        series_file_name = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(folder_path), series_IDs[0]
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_name)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        img = series_reader.Execute()
        volumeImg = VolumeImage(img, folder_path)
        return volumeImg

    def find_dicom_series_folder(self):
        if self.dicomSeriesFolder is not None:
            for folder in self.dirs:
                if self._is_a_dicom_series(folder):
                    print(f"- {folder.name} is a dicom series")
                    self.dicomSeriesFolder.append(folder)
                print(f"- {folder.name} is not a dicom series")
        print(
            f"found {len(self.dicomSeriesFolder)} dicom series already, "
            f"skip this time"
        )

    def delete_dicom_series(self):
        if self.dicomSeriesFolder is not None:
            self.dicomSeriesFolder = None
            print("delete dicom series")

    @staticmethod
    def _is_a_dicom_series(folder_path: STR_OR_PATH) -> bool:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder_path))
        if not series_IDs:
            print(
                f"ERROR: given directory {folder_path} does not "
                f"contain a DICOM series."
            )
            return False
        return True

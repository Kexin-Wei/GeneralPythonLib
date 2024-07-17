from pathlib import Path
from typing import Sequence, Optional, List, Union
from pprint import pprint

import SimpleITK as sitk
import pydicom
import pydicom.filereader

from .basic import FolderMg
from ..med_image.medical_image import VolumeImage, VolumeImageITK
from ..utils.define_class import STR_OR_PATH
from abc import ABC, abstractmethod


class MedicalImageFolderMgBase(FolderMg):
    """
    return images path given different image formats, currently supported
        - Meta Image: *.mha, *.mhd
        - Nifti Image: *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        - Nrrd Image: *.nrrd, *.nhdr
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__(folderFullPath)

    def get_nrrd_image_path(self) -> List[Path]:
        # *.nrrd, *.nhdr
        return self._get_file_path_by_extension_list(["nrrd", "nhdr"])

    def get_meta_image_path(self) -> List[Path]:
        # *.mha, *.mhd
        return self._get_file_path_by_extension_list(["mha", "mhd"])

    def get_nifti_image_path(self) -> List[Path]:
        # *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
        return self._get_file_path_by_extension_list(
            ["nia", "nii", "nii.gz", "hdr", "img", "img.gz"]
        )


FOLDERMG_OR_PATH_OR_STR = Union[FolderMg, Path, str]


class MedicalFolderMg(FolderMg):
    """
    Find certain file in a net-structure folder, which has multiple folders that contain their own folders inside them
    """

    def __init__(self, folderFullPath: STR_OR_PATH = Path()):
        super().__init__(folderFullPath)
        self.t2List = []
        self.dwiList = []
        self.adcList = []

    def search_T2_in_current_folder(self):
        if self.nFile:
            t2List = []
            for f in self.files:
                if "t2" in f.name.lower() and ("mha" in f.suffix or "nrrd" in f.suffix):
                    if "_cor" not in f.name.lower() and "_sag" not in f.name.lower():
                        t2List.append(f)
                        print(f)
            return t2List
        return []

    def search_dwi_in_current_folder(self):
        if self.nFile:
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
        if self.nFile:
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
        if self.nDirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_T2()
                self.t2List.extend(cMg.t2List)

    def get_DWI(self):
        self.dwiList.extend(self.search_dwi_in_current_folder())
        if self.nDirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_DWI()
                self.dwiList.extend(cMg.dwiList)

    def get_ADC(self):
        self.adcList.extend(self.search_adc_in_current_folder())
        if self.nDirs:
            for d in self.dirs:
                # print("\n--------------------------------------------")
                # print(f"In folder {d}")
                cMg = MedicalFolderMg(d)
                cMg.get_ADC()
                self.adcList.extend(cMg.adcList)


class DicomImageFolderMgBase(ABC, MedicalImageFolderMgBase):
    """
    Abstract base class for DICOM image folder management.
    """

    def __init__(self, folderFullPath: STR_OR_PATH, printOut: bool = False):
        super().__init__(folderFullPath)
        self.dicomSeriesFolder: Optional[Sequence[Path]] = None
        self.dicomSeries: Optional[Sequence[VolumeImage]] = None
        self.printOut = printOut
        self.read_all_dicom_series()

    @abstractmethod
    def read_all_dicom_series(self):
        """
        Read all DICOM series from a folder.

        Returns:
            Sequence[VolumeImageITK]: A list of DICOM series.
        """
        pass

    def get_all_dicom_series(self) -> Sequence[VolumeImage]:
        return self.dicomSeries


class DicomImageFolderMgITK(DicomImageFolderMgBase):
    """Add more dicom handling functions to BaseMedicalImageFolderMg

    Args:
        BaseMedicalImageFolderMg: return images path given different med_image formats, currently supported
            - Meta Image: *.mha, *.mhd
            - Nifti Image: *.nia, *.nii, *.nii.gz, *.hdr, *.img, *.img.gz
            - Nrrd Image: *.nrrd, *.nhdr

    """

    def __init__(self, folderFullPath: STR_OR_PATH, printOut: bool = False):
        super().__init__(folderFullPath, printOut=printOut)

    def read_all_dicom_series(self):
        """Read all dicom series from a folder

        Returns:
            Sequence[VolumeImage]: a list of dicom series
        """
        if not self._find_dicom_series_folder():
            return

        if self.printOut:
            print("read all dicom series")
        self.dicomSeries = []
        for dicomFolder in self.dicomSeriesFolder:
            self.dicomSeries.append(self._read_dicom_series(dicomFolder))
            if self.printOut:
                print(f"- {dicomFolder.name} read")

    def _find_dicom_series_folder(self) -> bool:
        if self.dicomSeriesFolder is not None:
            if self.printOut:
                print(
                    f"found {len(self.dicomSeriesFolder)} dicom series already, skip this time"
                )
            return True

        self.dicomSeriesFolder = []
        for folder in self.dirs:
            if self._is_a_dicom_series(folder):
                if self.printOut:
                    print(f"- {folder.name} is a dicom series")
                self.dicomSeriesFolder.append(folder)
            elif self.printOut:
                print(f"- {folder.name} is not a dicom series")
        if len(self.dicomSeriesFolder) == 0:
            if self.printOut:
                print("No dicom series found")
            return False
        return True

    def _read_dicom_series(self, folder_path: STR_OR_PATH) -> VolumeImageITK:
        """Read dicom series from a folder

        Args:
            folderPath (STR_OR_PATH): folder path

        Returns:
            Sequence[Path]: a list of dicom files
        """
        if not self._is_a_dicom_series(folder_path):
            return []

        volumeImg = VolumeImageITK()
        volumeImg.read(folder_path)
        return volumeImg

    def _is_a_dicom_series(self, folderPath: STR_OR_PATH) -> bool:
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folderPath))
        if not series_IDs and self.printOut:
            print(
                f"ERROR: given directory {folderPath} does not contain a DICOM series."
            )
            return False
        return True


class DicomImageFolderMg(DicomImageFolderMgBase):
    """
    Similar like DiacomImageFolderMgITK, but use pydicom to read dicom files
    """

    def __init__(self, folderFullPath: STR_OR_PATH):
        super().__init__(folderFullPath)

    def read_all_dicom_series(self):
        """Read all dicom series from a folder

        Returns:
            Sequence[VolumeImage]: a list of dicom series
        """
        for d in self.dirs:
            # read in pydicom
            try:
                dcm_dir = pydicom.dcmread(str(d))
            except Exception as e:
                print(f"Error reading dicom series: {e}")
                continue
            for patient_record in dcm_dir.patient_records:
                if hasattr(patient_record, "PatientID") and hasattr(
                    patient_record, "PatientsName"
                ):
                    print(
                        "Patient: {}: {}".format(
                            patient_record.PatientID, patient_record.PatientsName
                        )
                    )

                studies = patient_record.children
                # got through each serie
                for study in studies:
                    print(
                        " " * 4
                        + "Study {}: {}: {}".format(
                            study.StudyID, study.StudyDate, study.StudyDescription
                        )
                    )
                    all_series = study.children
                    # go through each serie
                    for series in all_series:
                        image_count = len(series.children)
                        plural = ("", "s")[image_count > 1]

                        # Write basic series info and image count

                        # Put N/A in if no Series Description
                        if "SeriesDescription" not in series:
                            series.SeriesDescription = "N/A"
                        print(
                            " " * 8
                            + "Series {}: {}: {} ({} image{})".format(
                                series.SeriesNumber,
                                series.Modality,
                                series.SeriesDescription,
                                image_count,
                                plural,
                            )
                        )

                        # Open and read something from each image, for demonstration
                        # purposes. For simple quick overview of DICOMDIR, leave the
                        # following out
                        print(" " * 12 + "Reading images...")
                        image_records = series.children
                        image_filenames = [
                            d.joinpath(image_rec.ReferencedFileID)
                            for image_rec in image_records
                        ]

                        datasets = [
                            pydicom.dcmread(image_filename)
                            for image_filename in image_filenames
                        ]

                        patient_names = set(ds.PatientName for ds in datasets)
                        patient_IDs = set(ds.PatientID for ds in datasets)

                        # List the image filenames
                        print("\n" + " " * 12 + "Image filenames:")
                        print(" " * 12, end=" ")
                        pprint(image_filenames, indent=12)

                        # Expect all images to have same patient name, id
                        # Show the set of all names, IDs found (should each have one)
                        print(
                            " " * 12
                            + "Patient Names in images..: {}".format(patient_names)
                        )
                        print(
                            " " * 12 + "Patient IDs in images..: {}".format(patient_IDs)
                        )
            if self.dicomSeries is None:
                self.dicomSeries = []
            self.dicomSeries.append(series)
            self.dicomSeriesFolder.append(d)

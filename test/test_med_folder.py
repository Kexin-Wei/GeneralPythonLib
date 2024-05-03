import pytest


def test_a_simple_sum():
    assert 1 + 1 == 2


def test_read_global(shared_datadir):
    from pathlib import Path

    print(shared_datadir)
    assert shared_datadir is not None


def test_read_dicom(shared_datadir):
    from lib.folder.med import DicomImageFolderMg

    dicom_folder = DicomImageFolderMg(shared_datadir.joinpath("dicom"))
    med_folder = DicomImageFolderMg(dicom_folder)
    for dcm in med_folder.dicomSeries:
        print(dcm)

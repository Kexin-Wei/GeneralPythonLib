import pytest


def test_a_simple_sum():
    assert 1 + 1 == 2


def test_read_global(shared_datadir):
    from pathlib import Path

    print(shared_datadir)
    assert shared_datadir is not None


def test_read_dicom(shared_datadir):
    from lib.folder.med import DicomImageFolderMg
    from lib.folder.basic import FolderMg
    import pydicom

    dicom_folder = FolderMg(str(shared_datadir.joinpath("dicom")))
    for dir in dicom_folder.dirs:
        try:
            a_dicom = pydicom.dcmread(dir)
            assert isinstance(a_dicom, pydicom.Dataset)
        except Exception as e:
            pytest.fail(f"Failed to read DICOM file: {dir}. Error: {str(e)}")

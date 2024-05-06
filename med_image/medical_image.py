import time
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from dataclasses import dataclass
from ..utils.define_class import STR_OR_PATH
from abc import ABC, abstractmethod


class VolumeImageInfo:
    """Volume med_image information for fast and simple access

    Attributes:
        image (sitk.Image): sitk image
        imageType (str): image type, e.g. MetaImage, NiftiImage, NrrdImage
        dimension (int): image dimension, e.g. 2, 3
        spacing (tuple): image spacing, e.g. (1.0, 1.0, 1.0)
        origin (tuple): image origin, e.g. (0.0, 0.0, 0.0)
        direction (tuple): image direction, e.g. (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        width (int): image width, e.g. 512
        height (int): image height, e.g. 512
        depth (int): image depth, e.g. 512
        pixelIDValue (int): pixel ID value, e.g. 8, 16, 32
        pixelIDType (str): pixel ID type, e.g. scalar, vector, offset
        pixelIDTypeAsString (str): pixel ID type as string, e.g. unsigned char, unsigned short, float

    """

    dimension: int
    spacing: tuple
    origin: tuple
    direction: tuple
    width: int
    height: int
    depth: int
    pixelIDValue: int
    pixelIDType: str
    pixelIDTypeAsString: str
    necessaryTagsValue: dict
    necessaryTags: dict = {
        "PatientName": "0010|0010",
        "PatientID": "0010|0020",
        "PatientBirthDate": "0010|0030",
        "StudyInstanceUID": "0020|000D",
        "StudyID": "0020|0010",
        "ImageType": "0008|0008",
        "StudyDate": "0008|0020",
        "StudyTime": "0008|0030",
        "AccessionNumber": "0008|0050",
        "Modality": "0008|0060",
        "SeriesDescription": "0008|103E",
    }


class VolumeImage(VolumeImageInfo):

    def __init__(self) -> None:
        super().__init__()
        self.image = None
        self.path = None
        self.imgInfo = VolumeImageInfo()

    def printOut(self):
        print(f"Image Path: {self.path}")
        print(f"Image Dimension: {self.imgInfo.dimension}")
        print(f"Image Spacing: {self.imgInfo.spacing}")
        print(f"Image Origin: {self.imgInfo.origin}")
        print(f"Image Direction: {self.imgInfo.direction}")
        print(f"Image Width: {self.imgInfo.width}")
        print(f"Image Height: {self.imgInfo.height}")
        print(f"Image Depth: {self.imgInfo.depth}")
        print(f"Image PixelIDValue: {self.imgInfo.pixelIDValue}")
        print(f"Image PixelIDType: {self.imgInfo.pixelIDType}")
        print(f"Image PixelIDTypeAsString: {self.imgInfo.pixelIDTypeAsString}")
        for k in self.imgInfo.necessaryTagsValue.keys():
            print(f"Image Necessary Tags {k}: {self.imgInfo.necessaryTagsValue[k]}")


class VolumeImageITK(VolumeImage):
    def __init__(self) -> None:
        super().__init__()
        self.series_file_name = None
        self.series_IDs = None
        self.metaData = None

    def read(self, folder_path: STR_OR_PATH):
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder_path))
        series_file_name = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(folder_path), series_IDs[0]
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_name)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        self.img = series_reader.Execute()
        self.path = folder_path
        series_reader.ReadImageInformation()
        self.imgInfo.dimension = self.img.GetDimension()
        self.imgInfo.spacing = self.img.GetSpacing()
        self.imgInfo.origin = self.img.GetOrigin()
        self.imgInfo.direction = self.img.GetDirection()
        self.imgInfo.width = self.img.GetWidth()
        self.imgInfo.height = self.img.GetHeight()
        self.imgInfo.depth = self.img.GetDepth()
        self.imgInfo.pixelIDValue = self.img.GetPixelIDValue()
        self.imgInfo.pixelIDType = self.img.GetPixelIDType()
        self.imgInfo.pixelIDTypeAsString = self.img.GetPixelIDTypeAsString()
        self.imgInfo.necessaryTagsValue = {}
        self.metaData = {}
        for meta_key in series_reader.GetMetaDataKeys():
            if meta_key in self.imgInfo.necessaryTags.keys():
                self.imgInfo.necessaryTagsValue[meta_key] = series_reader.GetMetaData(
                    meta_key
                )
            self.metaData[meta_key] = series_reader.GetMetaData(meta_key)

    def writeSlicesToDicom(self, outputPath: STR_OR_PATH):
        if not outputPath.exists():
            outputPath.mkdir(parents=True)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        for slice in range(self.depth):
            sliceImage = self.image[:, :, slice]
            sliceImage.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
            sliceImage.SetMetaData("0008|0013", time.strftime("%H%M%S"))
            writer.SetFileName(str(outputPath.joinpath(f"Slice{slice}.dcm")))
            writer.Execute(self.image[:, :, slice])
        print(f"write {self.depth} slices to {outputPath}")

    def writeToPng(self, outputPath: STR_OR_PATH, pngFormat=sitk.sitkUInt16):
        if not outputPath.exists():
            outputPath.mkdir(parents=True)
        sitk.WriteImage(
            sitk.Cast(sitk.RescaleIntensity(self.image), pngFormat),
            [outputPath.joinpath(f"slice{i}.png") for i in range(self.depth)],
        )

    def convertImageType(self, newImageType: sitk.Image) -> sitk.Image:
        return sitk.Cast(self.image, newImageType)

    def changeToUInt16(self, rescale: int = None) -> sitk.Image:
        newImageArray = sitk.GetArrayFromImage(self.image).astype(np.int64)
        for slice in range(self.depth):
            sliceImage = self.image[:, :, slice]
            sliceImageArray = sitk.GetArrayFromImage(sliceImage).astype(np.int64)
            sliceImageArray = sliceImageArray - sliceImageArray.min()
            if rescale is not None:
                sliceImageArray = sliceImageArray * rescale / sliceImageArray.max()
            sliceImageArray = sliceImageArray.astype(np.uint16)
            if slice == 0:
                print(
                    f"sliceImage Type:{sliceImage.GetPixelIDTypeAsString()}, slice type:{sliceImageArray.dtype}"
                )
            if sliceImageArray.max() > max:
                max = sliceImageArray.max()
            if sliceImageArray.min() < min:
                min = sliceImageArray.min()
            newImageArray[slice] = sliceImageArray

        newImage = sitk.GetImageFromArray(newImageArray)
        newImage.SetSpacing(self.spacing)
        newImage.SetOrigin(self.origin)
        newImage.SetDirection(self.direction)
        newImage.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        newImage.SetMetaData("0008|0013", time.strftime("%H%M%S"))
        # TODO metaData vs necessaryTagsValue
        # for key in self.metaData:
        # print(f"key:{key}, value:{reader.GetMetaData(slice, key)}")
        # sliceNewImage.SetMetaData(key, reader.GetMetaData(slice, key))
        return newImage

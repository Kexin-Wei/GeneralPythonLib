import numpy as np
import point_cloud_utils as pcu
from scipy.ndimage import binary_erosion


# overlap methods
def calc_dice_similarity_coefficient(label: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(label, prediction)
    return 2.0 * intersection.sum() / (label.sum() + prediction.sum())


def calc_jaccard_index(label: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(label, prediction)
    union = np.logical_or(label, prediction)
    return intersection.sum() / union.sum()


# distance methods
def get_surface_voxel_location(img: np.ndarray):
    eroded = binary_erosion(img)
    surface = img ^ eroded
    return np.argwhere(surface)


def calc_hausdorff_distance(label: np.ndarray, prediction: np.ndarray, spacing: np.ndarray):
    spacing = np.array(spacing)
    surface_label = get_surface_voxel_location(label) * spacing
    surface_prediction = get_surface_voxel_location(prediction) * spacing
    hausdorff = pcu.hausdorff_distance(surface_label, surface_prediction)
    return hausdorff


# volume methods
def calc_volumetric_similarity(label: np.ndarray, prediction: np.ndarray):
    return 1 - (np.abs(label.sum() - prediction.sum()) / (label.sum() + prediction.sum()))


def calc_relative_volume_difference(label: np.ndarray, prediction: np.ndarray):
    return np.abs(label.sum() - prediction.sum()) / label.sum()


# others
def calc_sensitivity(label: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(label, prediction)
    return intersection.sum() / label.sum()


def calc_specificity(label: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(label, prediction)
    return intersection.sum() / prediction.sum()


def calc_accuracy(label: np.ndarray, prediction: np.ndarray):
    intersection = np.logical_and(label, prediction)
    return intersection.sum() / label.sum()

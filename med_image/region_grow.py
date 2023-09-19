import numpy as np
import cv2 as cv
from collections import deque as Queue
from pathlib import Path
from ..utility.define_class import TwoDConnectionType


def connect_points(connect_type: TwoDConnectionType):
    if connect_type == TwoDConnectionType.eight:
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    else:  # connect_type == TwoDConnectionType.four:
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def location_matrx(i: int, j: int):
    x_m = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) + i
    y_m = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) + j
    return np.moveaxis(np.array([x_m, y_m]), 0, -1)


def intensity_matrix(img: np.ndarray, i: int, j: int):
    if i < 1 or j < 1 or i > img.shape[0] - 1 or j > img.shape[1] - 1:
        print("intensity_matrix: out of bound")
        return None
    return img[i - 1 : i + 2, j - 1 : j + 2]


def check_similarity(i_m: np.ndarray, c_m: np.ndarray, threshold=5):
    i_m_diff = np.abs(i_m - i_m[1][1])
    where_condition = np.multiply(i_m_diff < threshold, c_m)
    valid_loc = np.argwhere(where_condition)
    valid_i_m = np.where(
        where_condition,
        np.zeros_like(i_m_diff),
        np.ones_like(i_m_diff) * 255,
    )
    return valid_i_m, valid_loc


def get_new_seeds(
    img: np.ndarray, i: int, j: int, connect_type: TwoDConnectionType, threshold=5
):
    c_m = connect_points(connect_type)
    i_m = intensity_matrix(img, i, j)
    if i_m is None:
        return None, None
    seg_i_m, valid_loc = check_similarity(i_m, c_m, threshold=threshold)
    l_m = location_matrx(i, j)
    new_seeds_loc = l_m[valid_loc.T[0], valid_loc.T[1]]
    return seg_i_m, list(new_seeds_loc)


def region_growing(
    img: np.ndarray,
    init_seeds: Queue,
    connect_type: TwoDConnectionType,
    threshold=5,
):
    seeds = init_seeds
    record = []
    seg_img = np.zeros_like(img)
    iter = 0
    while seeds:
        iter += 1
        print(f"iter {iter}")
        if iter > 2e4:
            print("too many iterations")
            break

        p = seeds.popleft()
        i, j = p[0], p[1]
        record.append((i, j))

        seg_i_m, new_seed_loc = get_new_seeds(img, i, j, connect_type, threshold)

        if seg_i_m is None or new_seed_loc is None:
            continue

        for l in new_seed_loc:
            l = tuple(l)
            if l in seeds or l in record:
                continue
            seeds.append(l)

        old_seg_img = seg_img.copy()
        seg_img[i - 1 : i + 2, j - 1 : j + 2] = seg_i_m
        # if np.array_equal(old_seg_img, seg_img):
        #     break
    return seg_img


def fill_hole(img: np.ndarray) -> np.ndarray:
    """fill hole in image"""
    img = img.astype(np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    """normalize image"""
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    return img.astype(np.uint8)


def read_preprocess(file: Path) -> np.ndarray:
    """preprocess image"""
    img = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
    img = normalize(img)
    return img


def show_side_by_side(img1: np.ndarray, img2: np.ndarray):
    """show two images side by side"""
    new_img = np.concatenate((img1, img2), axis=1)
    cv.imshow("new_img", new_img)
    cv.waitKey(0)


def normalize(img: np.ndarray) -> np.ndarray:
    """normalize image"""
    img = img.astype(np.float32)
    img = img - img.min()
    img = img / img.max()
    img = img * 255
    return img.astype(np.uint8)


def read_preprocess(file: Path) -> np.ndarray:
    """preprocess image"""
    img = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
    img = normalize(img)
    return img


def show_side_by_side(img1: np.ndarray, img2: np.ndarray, input_point=None):
    """show two images side by side"""
    masked_img1 = cv.bitwise_and(img1, img1, mask=img2)
    new_img = np.concatenate((img1, masked_img1), axis=1)
    if input_point is None:
        cv.imshow("new_img", new_img)
        cv.waitKey(0)
        return
    assert len(input_point) == 2
    clr_img = cv.cvtColor(new_img, cv.COLOR_GRAY2BGR)
    cv.circle(clr_img, tuple(input_point), 3, (255, 0, 0), -1)
    cv.imshow("new_img", clr_img)
    cv.waitKey(0)


def show_img_from_file(file: Path) -> np.ndarray:
    img = cv.imread(str(file))
    cv.imshow("{file.name}", img)
    cv.waitKey(0)
    return img

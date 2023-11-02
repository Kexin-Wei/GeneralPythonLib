import numpy as np
import cv2 as cv
from collections import deque
from pathlib import Path
from ..utility.define_class import TwoDConnectionType, STR_OR_PATH
from enum import Enum


class Similarity(Enum):
    self_center = "seed it self"
    region_mean = "region mean"
    region_median = "region median"
    origin = "origin"


class RegionGrow:
    def __init__(
        self,
        img_file: str,
        prompt_point: tuple,
        threshold=5,
        connect_type: TwoDConnectionType = TwoDConnectionType.four,
        similarity_standard: Similarity = Similarity.origin,
        seg_rf_file: str = None,
    ):
        self.img = None
        self.seg_img = None
        self.img_file = Path(img_file)
        self.seg_rf_file = seg_rf_file
        self.connect_type = connect_type
        self.threshold = threshold
        self.c_m = None

        self.init_seeds = deque()
        self.similarity_standard = similarity_standard
        self.im_standard = np.nan

        if self.connect_type == TwoDConnectionType.eight:
            self.c_m = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        else:  # connect_type == TwoDConnectionType.four:
            self.c_m = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        self.img = cv.imread(str(self.img_file), cv.IMREAD_GRAYSCALE)
        assert (
            prompt_point[0] < self.img.shape[0]
            and prompt_point[1] < self.img.shape[1]
            and prompt_point[0] >= 0
            and prompt_point[1] >= 0
        ), "Prompt point is out of image"
        assert len(prompt_point) == 2
        self.prompt_point = prompt_point
        self.init_seeds.append(self.prompt_point)

        if self.seg_rf_file is not None:
            self.seg_rf_file = Path(seg_rf_file)
            self.seg_rf_img = cv.imread(str(self.seg_rf_file), cv.IMREAD_GRAYSCALE)
        self.normalize()

    def check_similarity(self, i_m: np.ndarray):
        masked_img = i_m * self.c_m
        if self.similarity_standard == Similarity.self_center:
            self.im_standard = i_m[1, 1]
        elif self.similarity_standard == Similarity.region_mean:
            self.im_standard = masked_img[masked_img > 0].mean()
        elif self.similarity_standard == Similarity.origin:
            self.im_standard = self.img[self.prompt_point[0], self.prompt_point[1]]
        else:
            # self.similarity_standard == Similarity.region_median:
            self.im_standard = np.median(masked_img[masked_img > 0])
        i_m_valid = np.abs(i_m - self.im_standard) < self.threshold
        where_condition = np.multiply(i_m_valid, self.c_m)
        valid_loc = np.argwhere(where_condition)
        return valid_loc

    def intensity_matrix(self, i: int, j: int):
        if i == 0 or j == 0 or i == self.img.shape[0] - 1 or j == self.img.shape[1] - 1:
            print(f"intensity_matrix at ({i},{j}) is out of bound")
            return None
        return self.img[i - 1 : i + 2, j - 1 : j + 2].T

    def get_new_seeds(self, i: int, j: int):
        i_m = self.intensity_matrix(i, j)
        if i_m is None:
            return None, None
        valid_loc = self.check_similarity(i_m)
        l_m = self.location_matrx(i, j)
        new_seeds_loc = l_m[valid_loc.T[0], valid_loc.T[1]]
        return list(new_seeds_loc)

    def region_growing(self):
        seeds = self.init_seeds
        record = []
        self.seg_img = np.zeros_like(self.img)
        iter = 0
        while seeds:
            iter += 1
            if iter > 2e4:
                print("too many iterations")
                break

            p = seeds.popleft()
            i, j = p[0], p[1]
            if iter > 1:
                assert self.im_standard != 0, f"error, {i,j}"

            if iter == 1:
                print(f"iter {iter}, seed: ({i},{j}), seeds len: {len(seeds)}")
            else:
                print(
                    f"iter {iter}, seed: ({i},{j}), seeds len: {len(seeds)}, standard {self.im_standard:.2f}"
                )
            record.append((i, j))
            self.seg_img[i, j] = 255

            new_seed_loc = self.get_new_seeds(i, j)
            if new_seed_loc is None:
                continue

            for l in new_seed_loc:
                l = tuple(l)
                if l in seeds or l in record:
                    continue
                if (
                    l[0] == 0
                    or l[1] == 0
                    or l[0] == self.img.shape[0] - 1
                    or l[1] == self.img.shape[1] - 1
                ):
                    continue
                seeds.append(l)
        self.seg_img = self.fill_hole(self.seg_img)
        return self.seg_img

    def fill_hole(self, img):
        """fill hole in image"""
        img = img.astype(np.uint8)
        return cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def normalize(self):
        """normalize image"""
        img = self.img
        img = img.astype(np.float32)
        img = img - img.min()
        img = img / img.max()
        img = img * 255
        self.img = img.astype(np.uint8)

    def show_side_by_side(self):
        """show two images side by side"""
        masked_img1 = cv.bitwise_and(self.img, self.img, mask=self.seg_img)
        new_img = np.concatenate((self.img, masked_img1, self.seg_img), axis=1)
        if self.prompt_point is None:
            cv.imshow("new_img", new_img)
            cv.waitKey(0)
            return
        self.show_prompt_point_in_image(new_img)

    def show_prompt_point_in_image(self, img: np.array):
        clr_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.circle(
            clr_img, (self.prompt_point[1], self.prompt_point[0]), 3, (0, 255, 0), -1
        )
        cv.imshow("new_img", clr_img)
        cv.waitKey(0)

    def show_prompt_point_at_start(self):
        if self.seg_rf_file is not None:
            self.show_prompt_point_in_image(self.seg_rf_img)
        else:
            self.show_prompt_point_in_image(self.img)

    @staticmethod
    def location_matrx(i: int, j: int):
        x_m = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) + i
        y_m = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) + j
        return np.moveaxis(np.array([x_m, y_m]), 0, -1)

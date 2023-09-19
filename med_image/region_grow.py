import numpy as np
import cv2 as cv
from collections import deque
from pathlib import Path
from ..utility.define_class import TwoDConnectionType, STR_OR_PATH


class RegionGrow:
    def __init__(
        self,
        img_file: str,
        prompt_point: tuple,
        threshold=5,
        connect_type: TwoDConnectionType = TwoDConnectionType.four,
    ):
        self.img = None
        self.seg_img = None
        self.img_file = Path(img_file)
        self.connect_type = connect_type
        self.threshold = threshold
        self.c_m = None
        self.prompt_point = prompt_point
        self.init_seeds = deque()

        if self.connect_type == TwoDConnectionType.eight:
            self.c_m = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        else:  # connect_type == TwoDConnectionType.four:
            self.c_m = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        self.init_seeds.append(self.prompt_point)

        self.img = cv.imread(str(self.img_file), cv.IMREAD_GRAYSCALE)
        self.normalize()

    def check_similarity(self, i_m: np.ndarray, threshold=5):
        i_m_diff = np.abs(i_m - i_m[1][1])
        where_condition = np.multiply(i_m_diff < threshold, self.c_m)
        valid_loc = np.argwhere(where_condition)
        valid_i_m = np.where(
            where_condition,
            np.zeros_like(i_m_diff),
            np.ones_like(i_m_diff) * 255,
        )
        return valid_i_m, valid_loc

    def intensity_matrix(self, i: int, j: int):
        if i < 1 or j < 1 or i > self.img.shape[0] - 1 or j > self.img.shape[1] - 1:
            print("intensity_matrix: out of bound")
            return None
        return self.img[i - 1 : i + 2, j - 1 : j + 2]

    def get_new_seeds(self, i: int, j: int):
        i_m = self.intensity_matrix(i, j)
        if i_m is None:
            return None, None
        seg_i_m, valid_loc = self.check_similarity(i_m)
        l_m = self.location_matrx(i, j)
        new_seeds_loc = l_m[valid_loc.T[0], valid_loc.T[1]]
        return seg_i_m, list(new_seeds_loc)

    def region_growing(self):
        seeds = self.init_seeds
        record = []
        self.seg_img = np.zeros_like(self.img)
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

            seg_i_m, new_seed_loc = self.get_new_seeds(i, j)
            if seg_i_m is None or new_seed_loc is None:
                continue

            for l in new_seed_loc:
                l = tuple(l)
                if l in seeds or l in record:
                    continue
                seeds.append(l)
            self.seg_img[i-1:i+2, j-1:j+2] = seg_i_m
        return self.seg_img

    def fill_hole(self):
        """fill hole in image"""
        img = self.img.astype(np.uint8)
        self.img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

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
        new_img = np.concatenate((self.img, masked_img1), axis=1)
        if self.prompt_point is None:
            cv.imshow("new_img", new_img)
            cv.waitKey(0)
            return
        assert len(self.prompt_point) == 2
        clr_img = cv.cvtColor(new_img, cv.COLOR_GRAY2BGR)
        cv.circle(clr_img, self.prompt_point, 3, (255, 0, 0), -1)
        cv.imshow("new_img", clr_img)
        cv.waitKey(0)

    @staticmethod
    def location_matrx(i: int, j: int):
        x_m = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) + i
        y_m = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) + j
        return np.moveaxis(np.array([x_m, y_m]), 0, -1)

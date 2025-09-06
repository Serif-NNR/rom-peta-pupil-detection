
import cv2
import math
from src.PupilDetector.Helpers.filters_and_methods import get_sobel_edge
import numpy as np


class PetaMetricsAndCorrection(object):

    def __init__(self):
        self.list_entropy, self.list_intensity, self.list_ellipse = [], [], []
        self.correction_data_interval = 120
    

    def get_entropy_and_intensity(self, feature_map, center, length, angle):
        center, length, angle = [int(center[0]), int(center[1])], [int(length[0]/2), int(length[1]/2)], int(angle)
        maxl = int(max(length))
        psx, pex, psy, pey = max(center[0] - maxl, 0), center[0] + maxl, max(center[1] - maxl, 0), center[1] + maxl
        map = feature_map[psy:pey, psx:pex]
        mask_center = (center[0] - psx, center[1] - psy)
        mask = np.zeros(map.shape)
        mask = cv2.ellipse(mask, mask_center, length, angle, 0, 360, 1, -1)
        mask = cv2.ellipse(mask, mask_center, length, angle, 0, 360, 0, 2)

        total_pixel_count = (mask > 0).sum()
        edge_map = get_sobel_edge(map)
        intensity_map, enropy_map = map * mask, edge_map * mask

        #print(mask.shape, feature_map.shape, length)
        intensity = np.sum(intensity_map[mask > 0]) / total_pixel_count
        entropy = np.sum(enropy_map[mask > 0]) / total_pixel_count
        return entropy, intensity


    def save_results(self, entropy, intensity, ellipse):
        self.list_entropy.append(entropy)
        self.list_intensity.append(intensity)
        self.list_ellipse.append(ellipse)


    def correct_ellipse_parameters(self, entropy, intensity, ellipse):
        if len(self.list_entropy) > self.correction_data_interval:
            self.list_entropy = self.list_entropy[-self.correction_data_interval: 0]
            self.list_intensity = self.list_intensity[-self.correction_data_interval: 0]

            if (math.dist(self.list_ellipse[-1][0], ellipse[0]) >= 19):  # LPW: 19, Dikablish: 52
                if (np.max(self.list_entropy) + max(np.std(self.list_entropy) ** 2, 1) <= entropy and
                    np.min(self.list_intensity) - max(np.std(self.list_intensity) ** 2, 1) >= intensity):
                    return self.list_ellipse[-1], self.list_intensity[-1], self.list_ellipse[-1]

        return entropy, intensity, ellipse



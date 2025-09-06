from src.PupilDetector.Helpers.constants import *
from src.PupilDetector.Helpers.filters_and_methods import filter_savitzky_golay, draw_rect_on_image
import numpy as np
from threading import Lock
import cv2

class TraditionalMethod(object):

    get_patch = None
    resolution_x, resolution_y = 0, 0
    pfx, pfy, thm = None, None, None

    def __init__(self, patch_type=Traditional_Patch_Type.EXTENDED, resolution=(320,240)):
        if patch_type == Traditional_Patch_Type.NORMAL: self.get_patch = self._get_patch
        elif patch_type == Traditional_Patch_Type.EXTENDED: self.get_patch = self._get_patch_extended
        self.resolution_x, self.resolution_y = resolution[0]-1, resolution[1]-1
        self.set_mtx = Lock()
        self.set_configuration(PF_Functions.find_pupil_area_med_first,
                               PF_Functions.find_pupil_area_med_first,
                               THM_Functions.define_thresholding8)

    def set_configuration(self, pfx, pfy, thm):
        with self.set_mtx:
            self.pfx = pfx
            self.pfy = pfy
            self.thm = thm

    def _get_patch(self, image):
        smim = self.get_smooth_minima_sequences(image)
        return self._find_pupil_patch_rectangle(smim)

    def _get_patch_extended(self, image):
        start_x, end_x, start_y, end_y, thx, thy, d = self._get_patch(image)
        end_y = end_y + 20 if end_y + 20 < self.resolution_y else self.resolution_y
        start_x = start_x - 20 if start_x - 20 >= 0 else 0
        start_y = start_y - 10 if start_y - 10 >= 0 else 0
        end_x = end_x + 10 if end_x + 10 < self.resolution_x else self.resolution_x
        return start_x, end_x, start_y, end_y, thx, thy, d

    def get_smooth_minima_sequences(self, image):
        minmax = [image[np.argmin(image, axis=0), np.arange(image.shape[1])],
                  image[np.arange(len(image)), np.argmin(image, axis=1)]]
        return filter_savitzky_golay(minmax)


    def find_pupil_patch_rectangle(self, d, pfx, pfy, thm):
        thx, thy = thm(d[0]), thm(d[1])
        try:
            start_x, end_x = pfx(d[0], thx)
            start_y, end_y = pfy(d[1], thy)
        except: start_x, start_y, end_x, end_y = 0, 0, 0, 0
        return start_x, end_x, start_y, end_y, thx, thy

    def _find_pupil_patch_rectangle(self, d):
        with self.set_mtx:
            thx, thy = self.thm(d[0]), self.thm(d[1])
            try:
                start_x, end_x = self.pfx(d[0], thx)
                start_y, end_y = self.pfy(d[1], thy)
            except: start_x, start_y, end_x, end_y = 0, 0, 0, 0
        return start_x, end_x, start_y, end_y, thx, thy, d


    def perform(self, image):
        return self.get_patch(image)


if __name__ == "__main__":
    image = cv2.imread("C:\\Users\\Sheriffnnr\\Desktop\\Fixtein PupilDetector\\image/1_1.avi_111_I.png")[:, :, 0]
    tm = TraditionalMethod()
    start_x, end_x, start_y, end_y, thx, thy, d = tm.perform(image)
    result = draw_rect_on_image((start_x, start_y), (end_x, end_y), d, image, thx, thy)

    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.show()

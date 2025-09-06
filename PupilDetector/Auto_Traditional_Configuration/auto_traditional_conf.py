from src.PupilDetector.Helpers.constants import Traditional_Patch_Type
from src.PupilDetector.Helpers.filters_and_methods import *
from src.PupilDetector.Learning_Model.learning_model import LearningModel
from src.PupilDetector.Traditional_Method.pf_function import PF_Functions
from src.PupilDetector.Traditional_Method.thm_function import THM_Functions
from src.PupilDetector.Traditional_Method.traditional_method import TraditionalMethod
from threading import Lock

class AutoTraditionalConfiguration(object):

    def __init__(self, traditional_method,
                 parameter_path, device_type, num_classes=1, input_channels=1,
                 patch_type=Traditional_Patch_Type.EXTENDED,
                 resolution=(320,240)):
        self.traditional_method = traditional_method
        self.tm = TraditionalMethod(patch_type, resolution)
        self.lm = LearningModel(device_type, num_classes, input_channels)
        self.list_pf, self.list_th = [], []
        self.base_thm = THM_Functions.define_thresholding8
        self.resolution = resolution[0] * resolution[1]
        self._init_func_list()
        self.set_mtx = Lock()

    def _init_func_list(self):
        self.list_pf = [
            PF_Functions.find_pupil_area_med_first,
            PF_Functions.find_pupil_area_max_length,
            PF_Functions.find_pupil_area_max_depth,
            PF_Functions.find_pupil_area_max_length_and_depth_product,
            PF_Functions.find_pupil_area_first_and_end
        ]
        self.list_th = [
            THM_Functions.define_thresholding1,
            THM_Functions.define_thresholding2,
            THM_Functions.define_thresholding4,
            THM_Functions.define_thresholding6,
            THM_Functions.define_thresholding7,
            THM_Functions.define_thresholding8,
            THM_Functions.define_thresholding9,
            THM_Functions.define_thresholding10
        ]

    def set_configuration(self, list_pf, list_th):
        with self.set_mtx:
            self.list_pf = self.list_pf
            self.list_th = self.list_th


    def perform(self, image):
        seg_map = self.lm.segment(image)
        smim = self.tm.get_smooth_minima_sequences(image)
        if len(self.list_th) == 0 or len(self.list_pf) == 0: return

        selected_pfx, selected_pfy, score = None, None, 0
        for pfx in self.list_pf:
            for pfy in self.list_pf:
                start_x, end_x, start_y, end_y, thx, thy = self.tm.find_pupil_patch_rectangle(smim, pfx, pfy, self.base_thm)
                size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / self.resolution)
                acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), seg_map)
                tscore = size_rate * min(acc)
                if tscore > score: selected_pfx, selected_pfy, score = pfx, pfy, tscore

        selected_thm, score = None, 0
        for thm in self.list_th:
            start_x, end_x, start_y, end_y, thx, thy = self.tm.find_pupil_patch_rectangle(smim, selected_pfx, selected_pfy, thm)
            size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / self.resolution)
            acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), map)
            tscore = size_rate * min(acc)
            if tscore > score: selected_thm, score = thm, tscore

        self.traditional_method.set_configuration(selected_pfx, selected_pfy, selected_thm)
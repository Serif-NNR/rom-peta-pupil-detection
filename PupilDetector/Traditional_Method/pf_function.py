import numpy as np

class PF_Functions(object):

    @staticmethod
    def find_pupil_area_med_first(d, th):
        mid_index = int(len(d) / 2)
        min_indexes = np.where(d < th)[0] - mid_index
        center_min_index = min_indexes[np.abs(min_indexes).argmin()] + mid_index
        border_end_indexes = np.where(d[center_min_index: len(d) - 1] >= th)[0]
        border_start_indexes = np.where(d[0: center_min_index + 1] >= th)[0]
        leftp, rightp = 0, len(d) - 1
        if len(border_start_indexes) != 0: leftp = border_start_indexes[-1]
        if len(border_end_indexes) != 0: rightp = border_end_indexes[0] + center_min_index
        return leftp, rightp

    @staticmethod
    def find_pupil_area_max_length(d, th):
        min_indexes = np.where(d < th)[0]
        splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
        if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
        i, selected_arr = 0, [0, 0]
        for arr in splited_mins:
            if len(arr) > i: i, selected_arr = len(arr), arr
        return selected_arr[0], selected_arr[-1]

    @staticmethod
    def find_pupil_area_max_depth(d, th):
        min_indexes = np.where(d < th)[0]
        splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
        if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
        i, selected_arr = 255, [0, 0]
        for arr in splited_mins:
            if len(arr) > 1:
                score = d[arr[0]: arr[-1]].min()
                if score < i: i, selected_arr = score, arr
        return selected_arr[0], selected_arr[-1]

    @staticmethod
    def find_pupil_area_max_length_and_depth_product(d, th):
        min_indexes = np.where(d < th)[0]
        splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
        if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
        i, selected_arr = 0, [0, 0]
        for arr in splited_mins:
            if len(arr) > 1:
                score = len(arr) * (th - d[arr[0]: arr[-1]].min())
                if score > i: i, selected_arr = score, arr
        return selected_arr[0], selected_arr[-1]

    @staticmethod
    def find_pupil_area_first_and_end(d, th):
        min_indexes = np.where(d < th)[0]
        if len(min_indexes) >= 2: return min_indexes[0], min_indexes[-1]
        else: return 0, 0



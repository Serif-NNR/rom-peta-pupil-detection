from src.PupilDetector.Traditional_Method.pf_function import PF_Functions
from src.PupilDetector.Traditional_Method.thm_function import THM_Functions
from enum import Enum


def get_detector_data():
    return {
        "timestamp_start": 0.0,
        "timestamp_end": 0.0,
        "start_x": 0.0,
        "end_x" : 0.0,
        "start_y": 0.0,
        "end_y": 0.0,
        "thx": 0.0,
        "thy": 0.0,
        "entropy": 0.0,
        "intensity": 0.0,
        "ellipse": ((0, 0), (0, 0), 0)
    }

class Traditional_Functions(object):
    @property
    def get_thresholding_functions(self):
        return {m: getattr(THM_Functions, m) for m in dir(THM_Functions) if not m.startswith('__')}

    @property
    def get_pupil_founder_functions(self):
        return {m: getattr(PF_Functions, m) for m in dir(PF_Functions) if not m.startswith('__')}


class Traditional_Patch_Type(Enum):
    NORMAL = 0,
    EXTENDED = 1

class Model_Device_Type(Enum):
    CPU = 0,
    CUDA = 1





if __name__ == "__main__":
    print(Traditional_Functions().get_thresholding_functions)
from fixein_traditional_method import *
import cv2
import glob

def get_ellipse_parameters(map):
    thxxx, threshed = cv2.threshold(map, 25, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedcnt, ellipse = None, [[0, 0], [0, 0], 0]
    cntcount = 0
    for cnt in cnts:
        if cnt.shape[0] > 5 and cntcount < cnt.shape[0]:
            cntcount = cnt.shape[0]
            selectedcnt = cnt
    if cntcount > 0: ellipse = cv2.fitEllipse(selectedcnt)
    return ellipse
def list_sort_key_dikablish(v):
    return int(v.split("\\")[-1].split("_")[3])
def get_dikablish_ss():
    list_ss = []
    images = glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\*_I.png')
    for image in images:
        name = image.split('\\')[-1].split('_')
        ss = name[0] + "_" + name[1] + "_" + name[2]
        if ss not in list_ss:
            list_ss.append(ss)
    return list_ss
ds = sorted(glob.glob("X:\\Pupil Dataset\\Dikablis_Full\\*_I.png"), key=list_sort_key_dikablish)


pf_method_dict = {"find_pupil_area_med_first": find_pupil_area_med_first,
             "find_pupil_area_max_length": find_pupil_area_max_length,
             "find_pupil_area_first_and_end": find_pupil_area_first_and_end,
             "find_pupil_area_max_depth": find_pupil_area_max_depth,
             "find_pupil_area_max_length_and_depth_product": find_pupil_area_max_length_and_depth_product}
th_method_dict = {"define_thresholding1": define_thresholding1,
             "define_thresholding2": define_thresholding2,
             "define_thresholding4": define_thresholding4,
             "define_thresholding6": define_thresholding6,
             "define_thresholding7": define_thresholding7,
             "define_thresholding8": define_thresholding8,
             "define_thresholding9": define_thresholding9,
             "define_thresholding10": define_thresholding10}

acc, rate, score, dikablish_conf_dict, ds = 0, 0, 0, dict(), 0
file = open("rom_dikablish_trad_conf.txt", "r")
line = file.readlines()

for l in line:
    part = l.split(";")
    nick = part[0].split(":")[1]
    nick_size = int(part[4].split(":")[1])
    ds += nick_size
    acc += (float(part[1].split(":")[1]) * nick_size)
    rate += (float(part[2].split(":")[1]) * nick_size)
    score += (float(part[3].split(":")[1]) * nick_size)
    #print(part)
    dikablish_conf_dict[nick + "pfx"] = pf_method_dict[part[5]]
    dikablish_conf_dict[nick + "pfy"] = pf_method_dict[part[6]]
    dikablish_conf_dict[nick + "thm"] = th_method_dict[part[7].replace("\n", "")]

print("dikablish-> acc:",acc/ds, " rate:",rate/ds, " score:",score/ds, ds)
#print(dikablish_conf_dict)


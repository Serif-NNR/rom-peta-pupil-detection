import glob, cv2
import numpy as np
from fixein_traditional_method import *
import threading

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





pf_method = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end,
             find_pupil_area_max_depth, find_pupil_area_max_length_and_depth_product]
th_method_base = define_thresholding8
th_method = [define_thresholding1, define_thresholding2,
             define_thresholding4, define_thresholding6,
             define_thresholding7, define_thresholding8, define_thresholding9,
             define_thresholding10]


dikablish_ss = get_dikablish_ss()
resolution = 320 * 240
ds_count, acc, err = 0, 0, 0
list_h, list_w, list_a = [], [], []




def perform_best_trad(ss_list):
    for ss in ss_list:
        new_ds = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format(ss)), key=list_sort_key_dikablish)

        analysis_ds = dict()
        for pfx in pf_method:
            for pfy in pf_method:
                for thm in th_method:
                    nick = pfx.__name__ + ";" + pfy.__name__ + ";" + thm.__name__
                    analysis_ds[nick + ";score"] = 0
                    analysis_ds[nick + ";acc"] = 0
                    analysis_ds[nick + ";rate"] = 0

        ds_count = 0
        for dsi in new_ds:
            image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
            minmax = procedure_min_max_simple_to_be_used(image)
            smim = filter_savitzky_golay_to_be_used(minmax)[0:2]

            for pfx in pf_method:
                for pfy in pf_method:
                    for thm in th_method:
                        nick = pfx.__name__ + ";" + pfy.__name__ + ";" + thm.__name__
                        start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, pfx, pfy, thm)
                        t_rate = (1 - (((end_y - start_y) * (end_x - start_x)) / resolution))
                        mask_acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), mask)
                        t_score = (t_rate * min(mask_acc))
                        min_acc = min(mask_acc)
                        t_acc = min_acc

                        analysis_ds[nick + ";score"] += t_score
                        analysis_ds[nick + ";acc"] += t_acc
                        analysis_ds[nick + ";rate"] += t_rate
            ds_count += 1

        max_score = 0
        max_nick = 0
        for pfx in pf_method:
            for pfy in pf_method:
                for thm in th_method:
                    nick = pfx.__name__ + ";" + pfy.__name__ + ";" + thm.__name__
                    if analysis_ds[nick + ";score"] > max_score:
                        max_score = analysis_ds[nick + ";score"]
                        max_nick = nick

        score = analysis_ds[max_nick + ";score"] / ds_count
        acc = analysis_ds[max_nick + ";acc"] / ds_count
        rate = analysis_ds[max_nick + ";rate"] / ds_count

        print("ss:{0}_;acc:{1};rate:{2};score:{3};ds:{4};{5}".format(ss, acc, rate, score, ds_count, max_nick))


ss_list = np.array(dikablish_ss)
ss_list = np.array_split(ss_list, 10)
th_list = []
for i in ss_list:
    th = threading.Thread(target=perform_best_trad, args=(i.tolist(),))
    th_list.append(th)
    th.start()
for i in th_list:
    i.join()


#perform_best_trad(dikablish_ss)

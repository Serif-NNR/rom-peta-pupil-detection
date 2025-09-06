from dataset_operation import DatasetAttributes
import pandas as pd
import cv2, time
import matplotlib.pyplot as plt
from fixein_traditional_method import *


ps_method = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end,
             find_pupil_area_max_depth, find_pupil_area_max_length_and_depth_product,]
th_method = [define_thresholding1, define_thresholding2,
             define_thresholding4, define_thresholding6,
             define_thresholding7, define_thresholding8, define_thresholding9,
             define_thresholding10]

ctg_analysis = dict()
def get_ctg_analysis_structure():
    category_dict = dict()
    for psmx in ps_method:
        category_dict[psmx.__name__] = dict()
        for psmy in ps_method:
            category_dict[psmx.__name__][psmy.__name__] = dict()
            for thm in th_method:
                category_dict[psmx.__name__][psmy.__name__][thm.__name__] = {
                    "avg_gain_size": 0,
                    "avg_pro_dur": 0,
                    "avg_pre_dur": 0,
                    "avg_pup_suc": 0,
                    "avg_std_suc": 0,

                    "image_size": list(),
                    "output_size": list(),
                    "gain_size": list(),
                    "process_duration_ms": list(),
                    "prepare_duration_ms": list(),
                    "output_pupil_success": list(),
                    "output_std_success": list(),

                    "output_success_rate": 0.4
                }
    return category_dict

if False:
    dataset = DatasetAttributes("P:\\LPW\\LPW_FULL\\", new_ds=True)
    dataset.Shortly(RATE=1)
    whole_set = pd.concat([dataset.train_df_full, dataset.val_df_full])


    for i, item in whole_set.iterrows():
        image_path, mask_path, category = item["image_path"], item["mask_path"], item["category"]
        image, mask = cv2.imread(image_path)[:, :, 0], cv2.imread(mask_path)[:, :, 0]
        if category not in ctg_analysis.keys(): ctg_analysis[category] = get_ctg_analysis_structure()
        #if category == "10_11": break

        ttime = time.time()
        minmax = procedure_min_max_simple_to_be_used(image)
        smim = filter_savitzky_golay_to_be_used(minmax)[0:2]
        prepare_duration_ms = time.time() - ttime

        for psmx in ps_method:
            for psmy in ps_method:
                for thm in th_method:

                    ttime = time.time()
                    start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, psmx, psmy, thm)
                    #start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, find_pupil_area_med_first, find_pupil_area_max_length, define_thresholding8)
                    process_duration_ms = time.time() - ttime

                    start_point, end_point = (start_x, start_y), (end_x, end_y)
                    result = draw_rect_on_image(start_point, end_point, smim, image, thx, thy)
                    acc = calculate_mask_acc(start_point, end_point, mask)
                    std_acc = 1 if acc[0] > 0.4 else 0

                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["prepare_duration_ms"].append(prepare_duration_ms)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["process_duration_ms"].append(process_duration_ms)
                    im_size, res_size = image.shape[0] * image.shape[1], (end_point[1] - start_point[1]) * (end_point[0] - start_point[0])
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["image_size"].append(im_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_size"].append(res_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["gain_size"].append(res_size / im_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_pupil_success"].append(acc)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_std_success"].append(std_acc)


                    if res_size / im_size < 0.08 and False:
                        print(res_size / im_size)
                        plt.imshow(result)
                        plt.show()

                    #break
                #break
            #break


    cur_time = str(time.time()).split(".")[0]
    for ctg in ctg_analysis.keys():
        for psx in ctg_analysis[ctg].keys():
            for psy in ctg_analysis[ctg][psx].keys():
                for th in ctg_analysis[ctg][psx][psy].keys():

                    ctg_analysis[ctg][psx][psy][th]["avg_gain_size"] = np.mean(ctg_analysis[ctg][psx][psy][th]["gain_size"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pro_dur"] = np.mean(ctg_analysis[ctg][psx][psy][th]["process_duration_ms"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pre_dur"] = np.mean(ctg_analysis[ctg][psx][psy][th]["prepare_duration_ms"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pup_suc"] = np.mean(ctg_analysis[ctg][psx][psy][th]["output_pupil_success"])
                    ctg_analysis[ctg][psx][psy][th]["avg_std_suc"] = np.mean(ctg_analysis[ctg][psx][psy][th]["output_std_success"])

                    ctg_analysis[ctg][psx][psy][th].pop("image_size")
                    ctg_analysis[ctg][psx][psy][th].pop("output_size")
                    ctg_analysis[ctg][psx][psy][th].pop("gain_size")
                    ctg_analysis[ctg][psx][psy][th].pop("process_duration_ms")
                    ctg_analysis[ctg][psx][psy][th].pop("prepare_duration_ms")
                    ctg_analysis[ctg][psx][psy][th].pop("output_pupil_success")
                    ctg_analysis[ctg][psx][psy][th].pop("output_std_success")

        df = pd.DataFrame.from_dict({(i, j, x): ctg_analysis[ctg][i][j][x]
                                for i in ctg_analysis[ctg].keys()
                                for j in ctg_analysis[ctg][i].keys()
                                for x in ctg_analysis[ctg][i][j].keys()},
                               orient='index')
        df.to_csv("TraditionalMethodAnalysis/{0}_CATEGORY-{1}.csv".format(cur_time, ctg))
else:
    dataset = DatasetAttributes("P:\\LPW\\LPW_FULL\\", new_ds=True)
    dataset.Shortly(RATE=1)
    whole_set = pd.concat([dataset.train_df_full, dataset.val_df_full])

    for i, item in whole_set.iterrows():
        image_path, mask_path, category = item["image_path"], item["mask_path"], item["category"]
        image, mask = cv2.imread(image_path)[:, :, 0], cv2.imread(mask_path)[:, :, 0]
        if category not in ctg_analysis.keys(): ctg_analysis[category] = get_ctg_analysis_structure()
        # if category == "10_11": break

        ttime = time.time()
        minmax = procedure_min_max_simple_to_be_used(image)
        smim = filter_savitzky_golay_to_be_used(minmax)[0:2]
        prepare_duration_ms = time.time() - ttime

        for psmx in ps_method:
            for psmy in ps_method:
                for thm in th_method:

                    ttime = time.time()
                    start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, psmx, psmy, thm)
                    # start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, find_pupil_area_med_first, find_pupil_area_max_length, define_thresholding8)
                    process_duration_ms = time.time() - ttime

                    start_point, end_point = (start_x, start_y), (end_x, end_y)
                    result = draw_rect_on_image(start_point, end_point, smim, image, thx, thy)
                    acc = calculate_mask_acc(start_point, end_point, mask)
                    std_acc = 1 if acc[0] > 0.4 else 0

                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["prepare_duration_ms"].append(
                        prepare_duration_ms)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["process_duration_ms"].append(
                        process_duration_ms)
                    im_size, res_size = image.shape[0] * image.shape[1], (end_point[1] - start_point[1]) * (
                                end_point[0] - start_point[0])
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["image_size"].append(im_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_size"].append(res_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["gain_size"].append(
                        res_size / im_size)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_pupil_success"].append(
                        acc)
                    ctg_analysis[category][psmx.__name__][psmy.__name__][thm.__name__]["output_std_success"].append(
                        std_acc)

                    if res_size / im_size < 0.08 and False:
                        print(res_size / im_size)
                        plt.imshow(result)
                        plt.show()

                    # break
                # break
            # break

    cur_time = str(time.time()).split(".")[0]
    for ctg in ctg_analysis.keys():
        for psx in ctg_analysis[ctg].keys():
            for psy in ctg_analysis[ctg][psx].keys():
                for th in ctg_analysis[ctg][psx][psy].keys():
                    ctg_analysis[ctg][psx][psy][th]["avg_gain_size"] = np.mean(
                        ctg_analysis[ctg][psx][psy][th]["gain_size"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pro_dur"] = np.mean(
                        ctg_analysis[ctg][psx][psy][th]["process_duration_ms"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pre_dur"] = np.mean(
                        ctg_analysis[ctg][psx][psy][th]["prepare_duration_ms"])
                    ctg_analysis[ctg][psx][psy][th]["avg_pup_suc"] = np.mean(
                        ctg_analysis[ctg][psx][psy][th]["output_pupil_success"])
                    ctg_analysis[ctg][psx][psy][th]["avg_std_suc"] = np.mean(
                        ctg_analysis[ctg][psx][psy][th]["output_std_success"])

                    ctg_analysis[ctg][psx][psy][th].pop("image_size")
                    ctg_analysis[ctg][psx][psy][th].pop("output_size")
                    ctg_analysis[ctg][psx][psy][th].pop("gain_size")
                    ctg_analysis[ctg][psx][psy][th].pop("process_duration_ms")
                    ctg_analysis[ctg][psx][psy][th].pop("prepare_duration_ms")
                    ctg_analysis[ctg][psx][psy][th].pop("output_pupil_success")
                    ctg_analysis[ctg][psx][psy][th].pop("output_std_success")

        df = pd.DataFrame.from_dict({(i, j, x): ctg_analysis[ctg][i][j][x]
                                     for i in ctg_analysis[ctg].keys()
                                     for j in ctg_analysis[ctg][i].keys()
                                     for x in ctg_analysis[ctg][i][j].keys()},
                                    orient='index')
        df.to_csv("TraditionalMethodAnalysis/{0}_CATEGORY-{1}.csv".format(cur_time, ctg))




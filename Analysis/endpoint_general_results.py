import glob
import math
import matplotlib.pyplot as plt
import os
import cv2
import skimage.exposure as exposure
from lpw_category import finetuning2
from fixein_traditional_method import *
import numpy as np
import torch
from tqdm import tqdm

com_learning =      [True]#[True, True, True, True, True, True]
com_traditional =   [False]#[False,False,True, True, True, True]
com_peta =          [True]#[False,True, False,False,True, True]
com_auto =          [False]#[False,False,False,True, False,True]
com_auto_duration = [12,    12,    40]

pf_method = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end,
             find_pupil_area_max_depth, find_pupil_area_max_length_and_depth_product]
th_method_base = define_thresholding8
th_method = [define_thresholding1, define_thresholding2,
             define_thresholding4, define_thresholding6,
             define_thresholding7, define_thresholding8, define_thresholding9,
             define_thresholding10]

dataset = [""]
device = "cuda:0"
traditional_configuration = None
auto_conf_duration_count = 1 * 40
resolution = 320 * 240
model_file = "X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt"
patch_model = None
fully_model = None
analysis = None
list_entropy, list_intensity, list_ellipse = [], [], []


def list_sort_key(v):
    return int(v.split("\\")[-1].split("_")[2])
def list_sort_key_dikablish(v):
    return int(v.split("\\")[-1].split("_")[3])

def get_trad_conf():
    return traditional_configuration

def set_trad_conf(conf):
    global traditional_configuration
    traditional_configuration = conf

def init_model():
    global patch_model, fully_model
    patch_model = torch.load(model_file).to(device)
    fully_model = torch.load(model_file).to(device)
    tns = torch.randn((1, 1, 320, 240)).to(device)
    for _ in range(3):
        with torch.no_grad(): _, _ = patch_model(tns), fully_model(tns)

def analysis_structure():
    return {
        "ltpa": [],
        "dataset": [],
        "image_count": 0,
        "patch_size": [],
        "containing_rate": [],
        "containing_success": [],
        "traditional_duration_list": [],
        "learning_duration_list" : [],
        "traditional_config_list": [],
        "full_duration_list": [],
        "x_px_error_list": [],
        "y_px_error_list": [],
        "center_point_px_error_list": [],
        "5px_error_list": [],
        "height_px_error_list": [],
        "width_px_error_list": [],
        "angle_px_error_list": []
    }


def get_sobel_edge(map):
    sobelx = cv2.Sobel(map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(map, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.multiply(sobelx, sobelx)
    sobely2 = cv2.multiply(sobely, sobely)
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    edge_im = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)).clip(0, 255).astype(
        np.uint8)
    return edge_im

def get_entropy_and_intensity(feature_map, center, length, angle):

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

    if False:
        f, axarr = plt.subplots(2, 3)
        axarr[0, 0].imshow(feature_map)
        axarr[0, 1].imshow(map)
        axarr[0, 2].imshow(mask)
        axarr[1, 0].imshow(edge_map)
        axarr[1, 1].imshow(intensity_map)
        axarr[1, 2].imshow(enropy_map)
        plt.show()

    #print(mask.shape, feature_map.shape, length)
    intensity = np.sum(intensity_map[mask > 0]) / total_pixel_count
    entropy = np.sum(enropy_map[mask > 0]) / total_pixel_count

    return entropy, intensity




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

def get_ellipse_parameters2(map):
    thxxx, threshed = cv2.threshold(map, 25, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedcnt, ellipse = None, [[0, 0], [0, 0], 0]
    cntcount = 0
    for cnt in cnts:
        if cnt.shape[0] < 10: continue
        tmp_ellipse = cv2.fitEllipse(cnt)
        if tmp_ellipse[1][0] > 6 and tmp_ellipse[1][1] > 6:
            entropy, intensity = get_entropy_and_intensity(map, tmp_ellipse[0], tmp_ellipse[1], tmp_ellipse[2])
            tmp_score = (255-entropy) * intensity
            if cnt.shape[0] > 5 and cntcount < tmp_score:
            #if cnt.shape[0] > 5 and cntcount < cnt.shape[0]:
                cntcount = tmp_score #cnt.shape[0]
                selectedcnt = cnt
    if cntcount > 0: ellipse = cv2.fitEllipse(selectedcnt)
    return ellipse

def perform_traditional_method(image, conf):
    minmax = procedure_min_max_simple_to_be_used(image)
    smim = filter_savitzky_golay_to_be_used(minmax)[0:2]
    start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, conf[0], conf[1], conf[2])

    if end_y - start_y < 10 and False:
        end_y = end_y + 10 if end_y + 10 < 240 else 239
        start_y = start_y - 10 if start_y - 10 >= 0 else 0
        end_x = end_x + 10 if end_x + 10 < 320 else 319
        start_x = start_x - 10 if start_x - 10 >= 0 else 0
    end_y = end_y + 20 if end_y + 20 < 240 else 239
    start_x = start_x - 20 if start_x - 20 >= 0 else 0
    start_y = start_y - 10 if start_y - 10 >= 0 else 0
    end_x = end_x + 10 if end_x + 10 < 320 else 319
    #end_x = end_x + 10 if end_x + 10 < 320 else 319
    #start_y = start_y - 10 if start_y - 10 >= 0 else 0
    return image[start_y:end_y, start_x:end_x], start_y, end_y, start_x, end_x, smim, thx, thy

def perform_detection(image):
    global patch_model
    with torch.no_grad(): map = patch_model(image.to(device))
    return map

def perform_auto(image):
    global fully_model, traditional_configuration
    temp = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
    temp = torch.tensor(temp.astype(np.float32) / 255.).unsqueeze(0)
    with torch.no_grad(): map = fully_model(temp.to(device))
    map = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(int)

    minmax = procedure_min_max_simple_to_be_used(image)
    smim = filter_savitzky_golay_to_be_used(minmax)[0:2]

    selected_pfx, selected_pfy, score = find_pupil_area_first_and_end, find_pupil_area_first_and_end, 0
    for pfx in pf_method:
        for pfy in pf_method:
                start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, pfx, pfy, th_method_base)
                size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / resolution)
                acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), map)
                tscore = size_rate * min(acc)
                if tscore > score: selected_pfx, selected_pfy, score = pfx, pfy, tscore

    selected_thm, score = define_thresholding8, 0
    for thm in th_method:
            start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, selected_pfx, selected_pfy, thm)
            size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / resolution)
            acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), map)
            tscore = size_rate * min(acc)
            if tscore > score: selected_thm, score = thm, tscore

    traditional_configuration = [selected_pfx, selected_pfy, selected_thm]

def perform_peta(map, ellipse):
    global list_entropy, list_intensity, list_ellipse
    entropy, intensity = get_entropy_and_intensity(map, ellipse[0], ellipse[1], ellipse[2])
    if len(list_entropy) > 120:
        last_center, last_size, temp_e, temp_i = list_ellipse[-1][0], list_ellipse[-1][1], list_entropy[-120: -1], list_intensity[-120:-1]
        low_w, high_w, low_h, high_h = last_size[0]*0.33, last_size[0]*3, last_size[1]*0.33, last_size[1]*3
        w_control = False if low_w <= ellipse[1][0] <= high_w else True
        h_control = False if low_h <= ellipse[1][1] <= high_h else True
        min_ang, max_ang, rec_ang = list_ellipse[-1][2] - 44, list_ellipse[-1][2] + 45, ellipse[2]
        angle_control = True if (min_ang <= rec_ang <= max_ang) \
            or (min_ang < 0 and (0 <= rec_ang <= max_ang or 179 + min_ang <= rec_ang)) \
            or (max_ang > 179 and (min_ang <= rec_ang <= 179 or rec_ang <= max_ang - 179)) else False

        #if(abs(last_center[0] - ellipse[0][0]) >= 35 or abs(last_center[1] - ellipse[0][1]) >= 26):
        #if(math.dist(last_center, ellipse[0]) > 26):
        if (math.dist(last_center, ellipse[0]) >= 19):  # LPW: 19, Dikablish: 52
            if (np.max(temp_e) + max(np.std(temp_e) ** 2, 1) <= entropy and
                np.min(temp_i) - max(np.std(temp_i) ** 2, 1) >= intensity):
                ellipse, entropy, intensity = list_ellipse[-1], list_entropy[-1], list_intensity[-1]
    list_entropy.append(entropy)
    list_intensity.append(intensity)
    return ellipse

def perform_peta_based_traditional(map, ellipse, norm_center):
    global list_entropy, list_intensity, list_ellipse
    entropy, intensity = get_entropy_and_intensity(map, ellipse[0], ellipse[1], ellipse[2])
    corrected = False
    if len(list_entropy) > 120:
        last_center, last_size, temp_e, temp_i = list_ellipse[-1][0], list_ellipse[-1][1], list_entropy[-120: -1], list_intensity[-120:-1]
        low_w, high_w, low_h, high_h = last_size[0] * 0.33, last_size[0] * 3, last_size[1] * 0.33, last_size[1] * 3
        w_control = False if low_w <= ellipse[1][0] <= high_w else True
        h_control = False if low_h <= ellipse[1][1] <= high_h else True
        #if(abs(last_center[0] - ellipse[0][0]) >= 35 or abs(last_center[1] - ellipse[0][1]) >= 26):
        if(math.dist(last_center, norm_center) >= 19  or w_control or h_control):  # LPW: 19, Dikablish: 52
            if (np.max(temp_e) + max(np.std(temp_e) ** 2, 1) <= entropy and
                np.min(temp_i) - max(np.std(temp_i) ** 2, 1) >= intensity):
                ellipse, entropy, intensity, corrected = list_ellipse[-1], list_entropy[-1], list_intensity[-1], True
    list_entropy.append(entropy)
    list_intensity.append(intensity)
    return ellipse, corrected









def general_results():
    global traditional_configuration, analysis, list_entropy, list_intensity, list_ellipse, auto_conf_duration_count
    init_model()

    for i in range(len(com_learning)):
        print("l:", com_learning[i], "t:", com_traditional[i], "p:", com_peta[i], "a:", com_auto[i])
        #auto_conf_duration_count = com_auto_duration[i]
        print(auto_conf_duration_count)
        analysis_ds = analysis_structure()
        traditional_configuration = [find_pupil_area_max_length, find_pupil_area_max_length, define_thresholding8]

        #dikablish_ss = get_dikablish_ss()
        #for ss in dikablish_ss:
        for ss in finetuning2.keys():
            list_mask = []
            list_calculated_x, list_calculated_y, list_annoted_x, list_annoted_y, \
            list_peta_x, list_peta_y= [], [], [], [], [], []
            #if ss != "DikablisT_20_3": continue
            print("\t[+] ", ss, ": ", end=" ")
            analysis_session = analysis_structure()
            analysis_session["ltpa"] = [com_learning[i], com_traditional[i], com_peta[i], com_auto[i]]
            start_y, end_y, start_x, end_x = 0, 0, 0, 0
            list_entropy, list_intensity, list_ellipse, list_ellipse_trad = [], [], [], []

            if com_auto[i] == False: traditional_configuration = finetuning2[ss] # For only LPW
            else: traditional_configuration = [find_pupil_area_max_length, find_pupil_area_max_length, define_thresholding8]

            images = sorted(glob.glob('P:\\LPW\\LPW_FULL\\{0}.avi*_I.png'.format(ss)), key=list_sort_key)
            #images = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format(ss)), key=list_sort_key_dikablish)

            image_counter = 0
            for image_path in images:
                mask_path = image_path.replace("_I.png", "_M.png")

                image, mask = cv2.imread(image_path)[:, :, 0], cv2.imread(mask_path)[:, :, 0]
                mask_ellipse = get_ellipse_parameters(mask)
                if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0) and len(list_mask) > 0: mask_ellipse = list_mask[-1]
                list_mask.append(mask_ellipse)

                if com_auto[i] and image_counter % auto_conf_duration_count == 0: perform_auto(image)
                if com_traditional[i]:
                    image, start_y, end_y, start_x, end_x, smim, thx, thy = perform_traditional_method(image, traditional_configuration)

                temp = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
                temp = torch.tensor(temp.astype(np.float32) / 255.).unsqueeze(0)
                map = perform_detection(temp)
                map = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)

                map_ellipse = get_ellipse_parameters(map)
                if (map_ellipse[0][0] == 0 or map_ellipse[0][1] == 0 or map_ellipse[1][0] < 5 or map_ellipse[1][1] < 5) \
                        and len(list_ellipse) > 0:
                    if com_traditional[i] == False: map_ellipse = list_ellipse[-1]
                    else: map_ellipse = list_ellipse_trad[-1]

                list_calculated_x.append(map_ellipse[0][0])
                list_calculated_y.append(map_ellipse[0][1])

                peta_not_suitable, peta_corrected = False, False
                if com_peta[i]:
                    try:
                        if com_traditional[i]:
                            temp_ellipse = list(map_ellipse)
                            temp_ellipse[0] = list(temp_ellipse[0])
                            temp_ellipse[0][0] += start_x
                            temp_ellipse[0][1] += start_y
                            map_ellipse, peta_corrected = perform_peta_based_traditional(map, map_ellipse, temp_ellipse[0])
                        else:
                            map_ellipse = perform_peta(map, map_ellipse)
                    except:
                        peta_not_suitable = True


                if com_traditional[i]: list_ellipse_trad.append(map_ellipse)

                if peta_not_suitable == False:
                    map_ellipse = list(map_ellipse)
                    map_ellipse[0] = list(map_ellipse[0])
                    if peta_corrected == False:
                        map_ellipse[0][0] += start_x
                        map_ellipse[0][1] += start_y
                    list_ellipse.append(map_ellipse)
                else:
                    map_ellipse = list(map_ellipse)
                    map_ellipse[0] = list(map_ellipse[0])
                    map_ellipse[0][0] += start_x
                    map_ellipse[0][1] += start_y
                    list_ellipse.append(map_ellipse)


                list_annoted_x.append(mask_ellipse[0][0])
                list_annoted_y.append(mask_ellipse[0][1])
                list_peta_x.append(map_ellipse[0][0])
                list_peta_y.append(map_ellipse[0][1])

                analysis_session["5px_error_list"].append(1 if math.dist(map_ellipse[0], mask_ellipse[0]) <= 5 else 0)
                analysis_session["center_point_px_error_list"].append(math.dist(map_ellipse[0], mask_ellipse[0]))
                analysis_session["x_px_error_list"].append(abs(map_ellipse[0][0]- mask_ellipse[0][0]))
                analysis_session["y_px_error_list"].append(abs(map_ellipse[0][1]- mask_ellipse[0][1]))
                analysis_session["height_px_error_list"].append(abs(map_ellipse[1][0]- mask_ellipse[1][0]))
                analysis_session["width_px_error_list"].append(abs(map_ellipse[1][1]- mask_ellipse[1][1]))
                analysis_session["angle_px_error_list"].append(abs(map_ellipse[2]- mask_ellipse[2]))
                analysis_session["patch_size"].append(((end_y - start_y) * (end_x - start_x)) / resolution)
                acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), mask)
                analysis_session["containing_rate"].append(acc[0])
                analysis_session["containing_success"].append(acc[1])

                image_counter += 1
                if analysis_session["5px_error_list"][-1] == 0 and False:
                    result = cv2.imread(image_path)#[:, :, 0]
                    #result = draw_rect_on_image((start_x, start_y), (end_x, end_y),smim, result, thx, thy)

                    #result[start_y:end_y, start_x:end_x, 2] = map
                    result[:, :, 2] = map
                    raw_predicted_image_cropped = cv2.ellipse(result, map_ellipse, 255, 1)
                    raw_predicted_image_cropped = cv2.circle(raw_predicted_image_cropped,
                                                             (int(map_ellipse[0][0]), int(map_ellipse[0][1])), 2,
                                                             (100, 100, 100), -1)
                    plt.imshow(raw_predicted_image_cropped)
                    tt = "!!! PETA !!! e:" if list_peta_x[-1] != list_calculated_x[-1] else "e:"
                    code = "_PETA_" if list_peta_x[-1] != list_calculated_x[-1] else "__"
                    plt.title(tt + str(list_entropy[-1]) + " ; i:" + str(list_intensity[-1]))
                    #plt.show()
                    cv2.imwrite("X:\\Fixein Pupil Detection\\general_results\\" + ss + '_' + str(image_counter) + code + str(list_entropy[-1]) + '_' + str(list_intensity[-1]) + ".png", raw_predicted_image_cropped)

            if False:
                line = np.arange(len(list_annoted_y))
                plt.plot(line, list_calculated_x, color='red', label='c_x')
                plt.plot(line, list_calculated_y, color='blue', label='c_y')
                plt.plot(line, list_annoted_x, color='yellow', label='a_x')
                plt.plot(line, list_annoted_y, color='black', label='a_y')
                plt.plot(line, list_peta_x, color='green', label='p_x')
                plt.plot(line, list_peta_y, color='purple', label='p_y')
                plt.legend()
                plt.show()

            print("5px err: ", np.asarray(analysis_session["5px_error_list"]).mean(), end="  ")
            print("x,y err: ", np.asarray(analysis_session["center_point_px_error_list"]).mean(), end="  ")
            print("x err: ", np.asarray(analysis_session["x_px_error_list"]).mean(), end="  ")
            print("y err: ", np.asarray(analysis_session["y_px_error_list"]).mean(), end="  ")
            print("h err: ", np.asarray(analysis_session["height_px_error_list"]).mean(), end="  ")
            print("w err: ", np.asarray(analysis_session["width_px_error_list"]).mean(), end="  ")
            print("a err: ", np.asarray(analysis_session["angle_px_error_list"]).mean(), end="  ")
            print("ps: ", np.asarray(analysis_session["patch_size"]).mean(), end="  ")
            print("cr: ", np.asarray(analysis_session["containing_rate"]).mean(), end="  ")
            print("cs: ", np.asarray(analysis_session["containing_success"]).mean())

            analysis_ds["5px_error_list"] += analysis_session["5px_error_list"]
            analysis_ds["center_point_px_error_list"] += analysis_session["center_point_px_error_list"]
            analysis_ds["x_px_error_list"] += analysis_session["x_px_error_list"]
            analysis_ds["y_px_error_list"] += analysis_session["y_px_error_list"]
            analysis_ds["height_px_error_list"] += analysis_session["height_px_error_list"]
            analysis_ds["width_px_error_list"] += analysis_session["width_px_error_list"]
            analysis_ds["angle_px_error_list"] += analysis_session["angle_px_error_list"]
            analysis_ds["patch_size"] += analysis_session["patch_size"]
            analysis_ds["containing_rate"] += analysis_session["containing_rate"]
            analysis_ds["containing_success"] += analysis_session["containing_success"]

        print("\n\n[+] FULL, l:", com_learning[i], "t:", com_traditional[i], "p:", com_peta[i], "a:", com_auto[i])
        print("5px err: ", np.asarray(analysis_ds["5px_error_list"]).mean())
        print("x,y err: ", np.asarray(analysis_ds["center_point_px_error_list"]).mean())
        print("x err: ", np.asarray(analysis_ds["x_px_error_list"]).mean())
        print("y err: ", np.asarray(analysis_ds["y_px_error_list"]).mean())
        print("h err: ", np.asarray(analysis_ds["height_px_error_list"]).mean())
        print("w err: ", np.asarray(analysis_ds["width_px_error_list"]).mean())
        print("a err: ", np.asarray(analysis_ds["angle_px_error_list"]).mean())
        print("ps: ", np.asarray(analysis_ds["patch_size"]).mean())
        print("cr: ", np.asarray(analysis_ds["containing_rate"]).mean())
        print("cs: ", np.asarray(analysis_ds["containing_success"]).mean())






def get_dikablish_ss():
    list_ss = []
    images = glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\*_I.png')
    for image in images:
        name = image.split('\\')[-1].split('_')
        ss = name[0] + "_" + name[1] + "_" + name[2]
        if ss not in list_ss:
            list_ss.append(ss)
    return list_ss


if __name__ == "__main__":
    general_results()
    #print(len(get_dikablish_ss()))
    #images = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format("dikablisR_1_1")), key=list_sort_key_dikablish)
    #print(images)





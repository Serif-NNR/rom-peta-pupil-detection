from dikablish_trad_conf import *
import numpy as np
import torch
import glob
import math
import matplotlib.pyplot as plt
import os
import cv2
import skimage.exposure as exposure
#model_file = "X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt"
device = "cuda:0"
traditional_configuration = None
auto_conf_duration_count = 1 * 40
pf_method = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end,
             find_pupil_area_max_depth, find_pupil_area_max_length_and_depth_product]
th_method_base = define_thresholding8
th_method = [define_thresholding1, define_thresholding2,
             define_thresholding4, define_thresholding6,
             define_thresholding7, define_thresholding8, define_thresholding9,
             define_thresholding10]
resolution = 320 * 240
list_entropy, list_intensity, list_ellipse, list_ellipse_trad = [], [], [], []


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
    end_y = end_y + 22 if end_y + 22 < 240 else 239
    start_x = start_x - 30 if start_x - 30 >= 0 else 0
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

    return [selected_pfx, selected_pfy, selected_thm]

def perform_peta(map, ellipse):
    global list_entropy, list_intensity, list_ellipse
    entropy, intensity = get_entropy_and_intensity(map, ellipse[0], ellipse[1], ellipse[2])
    if len(list_entropy) > 25:
        last_center, last_size, temp_e, temp_i = list_ellipse[-1][0], list_ellipse[-1][1], list_entropy[-25: -1], list_intensity[-25:-1]
        low_w, high_w, low_h, high_h = last_size[0]*0.33, last_size[0]*3, last_size[1]*0.33, last_size[1]*3
        w_control = False if low_w <= ellipse[1][0] <= high_w else True
        h_control = False if low_h <= ellipse[1][1] <= high_h else True
        min_ang, max_ang, rec_ang = list_ellipse[-1][2] - 44, list_ellipse[-1][2] + 45, ellipse[2]
        angle_control = True if (min_ang <= rec_ang <= max_ang) \
            or (min_ang < 0 and (0 <= rec_ang <= max_ang or 179 + min_ang <= rec_ang)) \
            or (max_ang > 179 and (min_ang <= rec_ang <= 179 or rec_ang <= max_ang - 179)) else False

        #if(abs(last_center[0] - ellipse[0][0]) >= 35 or abs(last_center[1] - ellipse[0][1]) >= 26):
        #if(math.dist(last_center, ellipse[0]) > 26):
        if (math.dist(last_center, ellipse[0]) >= 52):  # LPW: 19, Dikablish: 52
            if (np.max(temp_e) + max(np.std(temp_e) ** 2, 1) <= entropy and
                np.min(temp_i) - max(np.std(temp_i) ** 2, 1) >= intensity):
                ellipse, entropy, intensity = list_ellipse[-1], list_entropy[-1], list_intensity[-1]
    list_entropy.append(entropy)
    list_intensity.append(intensity)
    return ellipse
list_start_x, list_start_y, list_end_x, list_end_y = [], [], [], []
def perform_peta_based_traditional(map, ellipse, norm_center, start_x, start_y, end_x, end_y):
    global list_entropy, list_intensity, list_ellipse
    corrected = False
    try:entropy, intensity = get_entropy_and_intensity(map, ellipse[0], ellipse[1], ellipse[2])
    except: return ellipse, corrected, start_x, start_y, end_x, end_y
    fps_count = 25
    if len(list_entropy) > fps_count:
        last_center, last_size, temp_e, temp_i = list_ellipse[-1][0], list_ellipse[-1][1], list_entropy[-fps_count: -1], list_intensity[-fps_count:-1]
        low_w, high_w, low_h, high_h = last_size[0] * 0.33, last_size[0] * 3, last_size[1] * 0.33, last_size[1] * 3
        w_control = False if low_w <= ellipse[1][0] <= high_w else True
        h_control = False if low_h <= ellipse[1][1] <= high_h else True
        #if(abs(last_center[0] - ellipse[0][0]) >= 35 or abs(last_center[1] - ellipse[0][1]) >= 26):
        if(math.dist(last_center, norm_center) >= 52  or w_control or h_control):  # LPW: 19, Dikablish: 52
            if (np.max(temp_e) + max(np.std(temp_e) ** 2, 1) <= entropy and
                np.min(temp_i) - max(np.std(temp_i) ** 2, 1) >= intensity):
                ellipse, entropy, intensity, corrected = list_ellipse[-1], list_entropy[-1], list_intensity[-1], True
    list_entropy.append(entropy)
    list_intensity.append(intensity)
    if corrected: start_x, start_y, end_x, end_y = list_start_x[-1], list_start_y[-1], list_end_x[-1], list_end_y[-1]
    else:
        list_start_x.append(start_x)
        list_start_y.append(start_y)
        list_end_x.append(end_x)
        list_end_y.append(end_y)
    return ellipse, corrected, start_x, start_y, end_x, end_y

def init_model():
    global patch_model, fully_model
    patch_model = torch.load(model_file).to(device)
    fully_model = torch.load(model_file).to(device)
    tns = torch.randn((1, 1, 320, 240)).to(device)
    for _ in range(3):
        with torch.no_grad(): _, _ = patch_model(tns), fully_model(tns)

def ss_nick_dikablish(dsi):
    nick = dsi.split("\\")[-1].split("_")
    nick = nick[0] + "_" + nick[1] + "_" + nick[2] + "_"
    return nick


#model_file = "C:\\Users\\Sheriffnnr\\Desktop\\Fixtare Pupil Detection\\UnetPP_S\\UNetPP_FULL_FULL_FPD_25_l0.019_d0.962_vd0.894.pt"
#model_file = "C:\\Users\\Sheriffnnr\\Desktop\\Fixtare Pupil Detection\\UnetPP_S\\UNetPP-s_FULL_FPD_24_l0.019_d0.962_vd0.888.pt"
model_file = "X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt"
dikablish_ss = get_dikablish_ss()
acc, err, list_x, list_y, list_h, list_w, list_a = 0, 0, [], [], [], [], []
ds_count = 0
repred = 0
last_detection = 0
last_map = 0
previous_mask, previous_inference = None, None
patch_model = None
fully_model = None
init_model()
full_acc, full_size = 0, 0

for ss in dikablish_ss:
    new_ds = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format(ss)), key=list_sort_key_dikablish)
    nick = ss_nick_dikablish(new_ds[0])
    pfx, pfy, thm = dikablish_conf_dict[nick + "pfx"], dikablish_conf_dict[nick + "pfy"], dikablish_conf_dict[nick + "thm"]
    traditional_configuration = [pfx, pfy, thm]
    for dsi in new_ds:
        image2, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
        mask_ellipse = get_ellipse_parameters(mask)

        if ds_count % 4 == 0: traditional_configuration = perform_auto(image2) # 10Hz, 3Hz

        if True:
            image, start_y, end_y, start_x, end_x, smim, thx, thy = perform_traditional_method(image2, traditional_configuration)



        temp = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        temp = torch.tensor(temp.astype(np.float32) / 255.).unsqueeze(0)
        map = perform_detection(temp)
        petamap = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
        map = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
        #map[np.nonzero(map < 128)] = 0.0
        #map[np.nonzero(map >= 128)] = 255

        ellipse = get_ellipse_parameters(map)
        raw_ellipse = ellipse
        if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): mask_ellipse = previous_mask
        control = False
        if ((ellipse[0][0] == 0 and ellipse[0][1] == 0)  or ellipse[1][0] < 5 or ellipse[1][1] < 5):
            ellipse, control, raw_ellipse = previous_inference, True, last_detection

        if not control: new_center, last_map = (ellipse[0][0] + start_x, ellipse[0][1] + start_y), petamap
        else: new_center = ellipse[0]
        new_ellipse = (new_center, (ellipse[1][0], ellipse[1][1]), ellipse[2])


        ## PETA
        if True:
            map_ellipse, peta_corrected, start_x, start_y, end_x, end_y = perform_peta_based_traditional(last_map, raw_ellipse, new_center, start_x, start_y, end_x, end_y)
            if peta_corrected:
                print("PETA: [" + str(ds_count) + "]", math.dist(new_center, mask_ellipse[0]), "->", math.dist(map_ellipse[0], mask_ellipse[0]))
                ellipse = map_ellipse
                raw_ellipse = ((map_ellipse[0][0]-start_x, map_ellipse[0][1]-start_y), map_ellipse[1], map_ellipse[2])
                new_center = ellipse[0]

        t_rate = (1 - (((end_y - start_y) * (end_x - start_x)) / resolution))
        mask_acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), mask)
        min_acc = min(mask_acc)
        full_acc += min_acc
        full_size += t_rate


        px_err = math.dist(new_center, mask_ellipse[0])
        acc = acc + (1 if px_err <= 5 else 0)
        err = err + px_err
        list_h.append(abs(mask_ellipse[1][0] - ellipse[1][0]))
        list_w.append(abs(mask_ellipse[1][1] - ellipse[1][1]))
        list_a.append(abs(mask_ellipse[2] - ellipse[2]))
        list_x.append(abs(new_center[0] - mask_ellipse[0][0]))
        list_y.append(abs(new_center[1] - mask_ellipse[0][1]))

        if px_err > 5:
            print("[" + str(ds_count) + "]", dsi, px_err, "   ", err)
            if False and ds_count > 190:
                print("pred", new_ellipse)
                print("mask", mask_ellipse)
                plt.figure()
                f, axarr = plt.subplots(3, 1)
                pred = np.zeros((240, 320))
                pred[start_y:end_y, start_x:end_x] = map
                axarr[0].imshow(pred)
                axarr[1].imshow(mask)
                axarr[2].imshow(image2)
                plt.show()
        last_detection = raw_ellipse
        previous_mask = mask_ellipse
        previous_inference = new_ellipse
        list_ellipse.append(previous_inference)
        ds_count += 1


list_x, list_y, list_w, list_h, list_a = np.asarray(list_x), np.asarray(list_y), np.asarray(list_w), np.asarray(list_h), np.asarray(list_a)
print("repred: ", repred)
print("x", np.mean(list_x))
print("y", np.mean(list_y))
print("w", np.mean(list_w))
print("h", np.mean(list_h))
print("a", np.mean(list_a))
print("acc: ", full_acc / ds_count)
print("size: ", full_size, ds_count)
print("\n\nds_count: {0}\n"
      "acc: {1}\n"
      "norm acc: {2}\n"
      "err: {3}\n"
      "norm err: {4}\n".format(
    ds_count,
    acc, acc / ds_count, err, err / ds_count
))



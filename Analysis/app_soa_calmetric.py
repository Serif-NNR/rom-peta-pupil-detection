import glob, cv2, math, pandas as pd, numpy as np, time, torch, matplotlib.pyplot as plt, os
from fixein_traditional_method import *
from lpw_category import finetuning
from oursegmodel import SRO_UNet
from segmodel import *
from train_and_metric import dice_coef_metric

# DEFINE
PIXEL_ERROR_MAX = 21  # -1 index shows whole predictions which are higher than the pixel error in -2 index
RESOLUTION = (320, 240)
RESOLUTION_ORI = (640, 480)
RESOLUTION_DIV = (RESOLUTION[0] / RESOLUTION_ORI[0], RESOLUTION[1] / RESOLUTION_ORI[1])
TRIM_X, TRIM_Y = 0, 0
DEVICE = "cuda:0"
model = SRO_UNet(1, 1).to(DEVICE)
model.zero_grad()
modelPath = "16802016409633229_UNet_LPW_FULL_41R_SELECTED\\FPD_39_l0.054_d0.953_vd0.851.pt"
modelPath = "16804570461139164_SegNet_VGG16_R41\\FPD_34_l0.066_d0.945_vd0.776.pt"
modelPath = "1680597385_SRO_UNet_UandL1641858\\FPD_33_l0.087_d0.933_vd0.837.pt"
model = torch.load("ModelParameters\\" + modelPath)
start_t = str(time.time()).replace('.', '')
resultPath = "RawResultAddUp2\\" + modelPath.split('\\')[0] + "_" + start_t
os.mkdir(resultPath)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("MODEL NAME :  " + str(model))
print("MODEL PARM :  " + str(params))


def select_pupil_area(map):
    raw_predicted_image_cropped = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
    thxxx, threshed = cv2.threshold(raw_predicted_image_cropped, 20, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedcnt = None
    cntcount = 0
    for cnt in cnts:
        if cnt.shape[0] > 5 and cntcount < cnt.shape[0]:
            cntcount = cnt.shape[0]
            selectedcnt = cnt
    if cntcount > 0:
        ellipse = cv2.fitEllipse(selectedcnt)
        raw_predicted_image_cropped = cv2.ellipse(raw_predicted_image_cropped, ellipse, 255, 10)
        raw_predicted_image_cropped = cv2.circle(raw_predicted_image_cropped, (int(ellipse[0][0]), int(ellipse[0][1])),
                                                 4, (40, 40, 40), -1)
    return cntcount > 0, cv2.fitEllipse(selectedcnt) if cntcount > 0 else None, raw_predicted_image_cropped


def perform_learning_model(im):
    pred = np.expand_dims(im, axis=-1).transpose((2, 0, 1))
    pred_dur = time.time()
    pred = torch.tensor(pred.astype(np.float32) / 255.).unsqueeze(0)
    pred = model(pred.to(DEVICE))
    return pred, time.time() - pred_dur


def perform_traditional_method(image, mask, category):
    pd, th = finetuning[category.replace("#", "")][0], finetuning[category.replace("#", "")][1]
    t_duraton = time.time()
    minmax = procedure_min_max_simple(image)
    smim = filter_savitzky_golay(minmax)
    start_x, end_x, start_y, end_y, thx, thy = find_pupil_area(smim[0:2], pd, th)
    if start_x - end_x < 30:
        start_x -= 15
        end_x += 15
        if start_x < 0: start_x = 0
        if end_x > RESOLUTION[0]: end_x = RESOLUTION[0] - 1
    if start_y - end_y < 30:
        start_y -= 15
        end_y += 15
        if start_y < 0: start_y = 0
        if end_y > RESOLUTION[1]: end_y = RESOLUTION[1] - 1
    t_duraton = time.time() - t_duraton
    start_point = (start_x, start_y)
    end_point = (end_x, end_y)
    result = draw_rect_on_image(start_point, end_point,
                                smim, image, thx, thy)
    resmask = draw_rect_on_image(start_point, end_point,
                                 smim, mask, thx, thy)
    acc = calculate_mask_acc(start_point, end_point, mask)
    cropped_image = image[start_y:end_y, start_x:end_x]
    return start_x, end_x, start_y, end_y, result, resmask, acc, t_duraton


def perform_stochastic_rationale_outcome(adict):
    tsx, tex, tsy, tey, adict["analysis_image_base"], adict["analysis_mask_base"], t_acc, t_dur = \
        perform_traditional_method(adict["image"], adict["mask"], adict["category"])
    adict["traditional_accuracy"].append(t_acc)
    adict["traditional_method_duration"].append(t_dur)

    adict["traditional_map"], t_duration = perform_learning_model(adict["image"][tsy:tey + 1, tsx:tex + 1])
    adict["normal_map"], n_duration = perform_learning_model(adict["image"])
    adict["traditional_prediction_duration"].append(t_duration)
    adict["normal_prediction_duration"].append(n_duration)

    adict["traditional_ellipse_found"], adict["traditional_ellipse"], adict["traditional_segmentation_final_map"] = \
        select_pupil_area(adict["traditional_map"])
    adict["normal_ellipse_found"], adict["normal_ellipse"], adict["normal_segmentation_final_map"] = \
        select_pupil_area(adict["normal_map"])

    adict["traditional_patch_start_point"] = (tsx, tsy)
    adict["traditional_mask"] = adict["mask"][tsy:tey + 1, tsx:tex + 1]

    return adict


def analysis_dictionary():
    return {
        # Sample Specific Attributes
        "category": "",
        "total_count": 0,
        "image": None,
        "mask": None,
        "traditional_mask": None,
        "traditional_map": None,
        "normal_map": None,
        "traditional_patch_start_point": None,
        "traditional_ellipse:": None,
        "normal_ellipse": None,
        "traditional_ellipse_found": False,
        "normal_ellipse_found": False,
        "analysis_image_base": None,
        "analysis_mask_base": None,
        "traditional_segmentation_final_map": None,
        "normal_segmentation_final_map": None,

        # General Attributes
        "traditional_accuracy": list(),
        "traditional_prediction_duration": list(),
        "traditional_method_duration": list(),
        "normal_prediction_duration": list(),
        "traditional_IoU": list(),
        "traditional_IoU_full": list(),
        "normal_IoU": list(),
        "traditional_pixel_error": [0 for i in range(0, PIXEL_ERROR_MAX)],
        "traditional_pixel_error_full": [0 for i in range(0, PIXEL_ERROR_MAX)],
        "normal_pixel_error": [0 for i in range(0, PIXEL_ERROR_MAX)]
    }


def perform_manual_garbage_collector(adict):
    adict["image"]= None
    adict["mask"]= None
    adict["traditional_mask"]= None
    adict["traditional_map"]= None
    adict["normal_map"]= None
    adict["traditional_patch_start_point"]= None
    adict["traditional_ellipse:"]= None
    adict["normal_ellipse"]= None
    adict["traditional_ellipse_found"]= False
    adict["normal_ellipse_found"]= False
    adict["analysis_image_base"]= None
    adict["analysis_mask_base"]= None
    adict["traditional_segmentation_final_map"]= None
    adict["normal_segmentation_final_map"]= None
    adict["traditional_accuracy"] = None
    return adict


def get_total_analysis_results(analysis_dict):
    all_result = analysis_dictionary()
    all_result["category"] = "ALL"
    def get_average(adict, topic):
        total, cnt = 0, 0
        for v in adict.values():
            total, cnt = total + v[topic], cnt + 1
        return total / cnt
    all_result["traditional_pixel_error"] = get_average(analysis_dict, "traditional_pixel_error")
    all_result["traditional_pixel_error_full"] = get_average(analysis_dict, "traditional_pixel_error_full")
    all_result["normal_pixel_error"] = get_average(analysis_dict, "normal_pixel_error")
    all_result["traditional_method_duration"] = get_average(analysis_dict, "traditional_method_duration")
    all_result["normal_prediction_duration"] = get_average(analysis_dict, "normal_prediction_duration")
    all_result["traditional_prediction_duration"] = get_average(analysis_dict, "traditional_prediction_duration")
    all_result["traditional_IoU"] = get_average(analysis_dict, "traditional_IoU")
    all_result["traditional_IoU_full"] = get_average(analysis_dict, "traditional_IoU_full")
    all_result["normal_IoU"] = get_average(analysis_dict, "normal_IoU")
    return all_result


def analysis_visualization(analysis_dict):
    def norm_pixel_error(pel):
        for i in range(1, PIXEL_ERROR_MAX):
            pel[i] += pel[i-1]
        return pel
    analysis_dict["traditional_pixel_error"] = np.asarray(norm_pixel_error(analysis_dict["traditional_pixel_error"])) / analysis_dict["total_count"]
    analysis_dict["traditional_pixel_error_full"] = np.asarray(norm_pixel_error(analysis_dict["traditional_pixel_error_full"])) / analysis_dict["total_count"]
    analysis_dict["normal_pixel_error"] = np.asarray(norm_pixel_error(analysis_dict["normal_pixel_error"])) / analysis_dict["total_count"]
    analysis_dict["traditional_method_duration"] = np.average(analysis_dict["traditional_method_duration"])
    analysis_dict["normal_prediction_duration"] = np.average(analysis_dict["normal_prediction_duration"])
    analysis_dict["traditional_prediction_duration"] = np.average(analysis_dict["traditional_prediction_duration"])
    analysis_dict["traditional_IoU"] = np.average(analysis_dict["traditional_IoU"])
    analysis_dict["traditional_IoU_full"] = np.average(analysis_dict["traditional_IoU_full"])
    analysis_dict["normal_IoU"] = np.average(analysis_dict["normal_IoU"])
    return analysis_dict


def calculate_pixel_error(p1, p2):
    error_rate = math.dist(p1, p2)
    return int(error_rate) if error_rate < PIXEL_ERROR_MAX else PIXEL_ERROR_MAX-1


def analysis_results(cp, adict):
    adict["total_count"] += 1
    tmp, nmp, tsp = adict["traditional_map"], adict["normal_map"], adict["traditional_patch_start_point"]
    mask, t_mask = adict["mask"], adict["traditional_mask"]
    detect_rate_t_mask = (mask[tsp[-1]:tsp[-1] + tmp.shape[-2], tsp[-2]:tsp[-2] + tmp.shape[-1]] > 125).sum()
    detect_rate_mask = (mask > 125).sum()

    t_norm_map = np.zeros((nmp.shape[-2], nmp.shape[-1]))
    t_norm_map[tsp[-1]:tsp[-1]+tmp.shape[-2], tsp[-2]:tsp[-2]+tmp.shape[-1]] = tmp.detach().cpu().numpy()[0, 0, :, :]
    tfull_control = True if detect_rate_t_mask > 5 else False
    normal_control = True if detect_rate_mask > 5 else False

    adict["traditional_IoU"].append(dice_coef_metric(tmp.detach().cpu().numpy()[0, 0, :, :], t_mask))
    adict["traditional_IoU_full"].append(dice_coef_metric(t_norm_map, mask))
    adict["normal_IoU"].append(dice_coef_metric(nmp.detach().cpu().numpy()[0, 0, :, :], mask))

    if tfull_control:
        if adict["traditional_ellipse"] == None: adict["traditional_pixel_error"][PIXEL_ERROR_MAX-1] += 1
        else: adict["traditional_pixel_error"][calculate_pixel_error((adict["traditional_ellipse"][0][0] + tsp[0],
                                               adict["traditional_ellipse"][0][1] + tsp[1]), cp)] += 1
    else: adict["traditional_pixel_error"][0] += 1
    if normal_control :
        if adict["traditional_ellipse"] == None: adict["traditional_pixel_error_full"][PIXEL_ERROR_MAX-1] += 1
        else: adict["traditional_pixel_error_full"][calculate_pixel_error((adict["traditional_ellipse"][0][0] + tsp[0],
                                                    adict["traditional_ellipse"][0][1] + tsp[1]), cp)] += 1
    else: adict["traditional_pixel_error_full"][0] += 1
    if normal_control:
        if adict["normal_ellipse"] == None: adict["normal_pixel_error"][PIXEL_ERROR_MAX-1] += 1
        else: adict["normal_pixel_error"][calculate_pixel_error(adict["normal_ellipse"][0], cp)] += 1
    else: adict["normal_pixel_error"][0] += 1
    return adict


def get_center_point(f_name, cnt):
    f = open(f_name, "r")
    l = f.readlines()
    cp = l[cnt].split(" ")
    f.close()
    return (int(float(cp[0]) * RESOLUTION_DIV[0]), int(float(cp[1]) * RESOLUTION_DIV[1]))


def save_as_video(resultVideo, resultVideoCropped, cp, adict, normal_result, traditional_result):
    tmap = cv2.circle(traditional_result, (cp[0], cp[1]), 1, (255, 0, 255), -1)
    nmap = cv2.circle(normal_result, (cp[0], cp[1]), 1, (255, 0, 255), -1)
    color, thx, size, font = (0, 0, 0), 1, 0.35, "" # ImageFont.truetype("Roboto-Regular.ttf", 50)
    t_fps = 1 / (adict["traditional_prediction_duration"][-1] + adict["traditional_method_duration"][-1])
    n_fps = 1 / adict["normal_prediction_duration"][-1]
    t_cp = (adict["traditional_ellipse"][0][0] + adict["traditional_patch_start_point"][0],
            adict["traditional_ellipse"][0][1] + adict["traditional_patch_start_point"][1])
    n_cp = adict["normal_ellipse"][0]

    tmap = cv2.putText(tmap, "Diff   : " + str(calculate_pixel_error(cp, t_cp)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    tmap = cv2.putText(tmap, "t Dur : {:.4f}".format(adict["traditional_method_duration"][-1]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    tmap = cv2.putText(tmap, "p Dur : {:.4f}".format(adict["traditional_prediction_duration"][-1]), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    tmap = cv2.putText(tmap, "FPS  : {:.1f}".format(t_fps), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)

    nmap = cv2.putText(nmap, "Diff   : " + str(calculate_pixel_error(cp, n_cp)), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    nmap = cv2.putText(nmap, "t Dur : {:.4f}".format(adict["traditional_method_duration"][-1]), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    nmap = cv2.putText(nmap, "p Dur : {:.4f}".format(adict["normal_prediction_duration"][-1]), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)
    nmap = cv2.putText(nmap, "FPS  : {:.1f} ".format(n_fps), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, size, color, thx)

    resultVideo.write(nmap)
    resultVideoCropped.write(tmap)


def analysis_base(df_dict):
    resultVideo = cv2.VideoWriter(resultPath+"\\" + df_dict["category"] + "_" + start_t + ".avi", cv2.VideoWriter_fourcc(*'MJPG'), 60, RESOLUTION)
    resultVideoCropped = cv2.VideoWriter(resultPath + "\\" + df_dict["category"] + "_" + start_t + "_CROPPED.avi", cv2.VideoWriter_fourcc(*'MJPG'), 60, RESOLUTION)
    analysis_dict = analysis_dictionary()
    analysis_dict["category"] = df_dict["category"]
    image_video, mask_video = cv2.VideoCapture(df_dict["image_path"]), cv2.VideoCapture(df_dict["annot_path"])
    success, im = image_video.read()
    _, msk = mask_video.read()
    temp_n_ell, temp_t_ell, temp_tsp = [[0, 0]], [[0, 0]], [0, 0]
    while success:
        im1, msk1, cp = im[:, :, 0], msk[:, :, 0], get_center_point(df_dict["center_point"], analysis_dict["total_count"])
        im1, msk1 = cv2.resize(im1, RESOLUTION, cv2.INTER_AREA), cv2.resize(msk1, RESOLUTION, cv2.INTER_AREA)
        analysis_dict["image"], analysis_dict["mask"] = im1, msk1
        # tmp, nmp, tsp, t_ell, n_ell, twp, nwp, tdr, ndr, t_msk, t_res, t_ind, n_ind, t_dur
        analysis_dict = perform_stochastic_rationale_outcome(analysis_dict)
        #analysis_dict = analysis_results(tmp, nmp, msk1, t_msk, tsp, t_ell, n_ell, cp, tdr, ndr, analysis_dict)
        analysis_dict = analysis_results(cp, analysis_dict)

        # We place segmentation maps to analysis image base. Also, analysis mask base can be used for this purpose
        t_res, t_res2 = analysis_dict["analysis_image_base"].copy(), analysis_dict["analysis_image_base"].copy()
        t_ind, tsp = analysis_dict["traditional_segmentation_final_map"], analysis_dict["traditional_patch_start_point"]
        t_res[:, :, 2] = analysis_dict["normal_segmentation_final_map"]
        t_res2[tsp[-1]:tsp[-1] + t_ind.shape[-2], tsp[-2]:tsp[-2] + t_ind.shape[-1], 2] = t_ind

        # If we can't calculate an ellipse, we'll use the just previous center points
        if analysis_dict["normal_ellipse"] == None: analysis_dict["normal_ellipse"] = temp_n_ell
        else: temp_n_ell = analysis_dict["normal_ellipse"]
        if analysis_dict["traditional_ellipse"] == None: analysis_dict["traditional_ellipse"] = temp_t_ell
        else: temp_t_ell = analysis_dict["traditional_ellipse"]
        if analysis_dict["traditional_patch_start_point"] == None: analysis_dict["traditional_patch_start_point"] = temp_tsp
        else: temp_tsp = analysis_dict["traditional_patch_start_point"]

        save_as_video(resultVideo, resultVideoCropped, cp, analysis_dict, t_res, t_res2)
        success, im = image_video.read()
        _, msk = mask_video.read()
    analysis_dict = analysis_visualization(analysis_dict)
    analysis_dict = perform_manual_garbage_collector(analysis_dict)
    resultVideo.release()
    resultVideoCropped.release()
    return analysis_dict


def seperation_lpw(file, type=0):
    file_sep = file.split("\\")
    if type == 0:
        return file_sep[-2] + "_" + file_sep[-1]
    elif type == 1:
        return file.split(".")[0] + ".txt"
    elif type == 2:
        f = open("P:\\LPW\\fileassignement.txt", "r")
        for l in f.readlines():
            if file_sep[-2] + "/" + file_sep[-1] + " " == l.split("->")[0]:
                r = "P:\\LPW\\ANNOTATIONS\\" + l.split("->")[1].split('/')[1].split('\n')[0] + "pupil_seg_2D.mp4"
                f.close()
                return r
    return ""


def get_dataset(path, seperation=seperation_lpw, annot=False):
    image_file = glob.glob(path + "\\*\\*.avi")
    df = pd.DataFrame({
        "image_path": [im for im in image_file],
        "annot_path": [seperation(im, type=2) for im in image_file],
        "category": ["#" + seperation(im, type=0) for im in image_file],
        "center_point": [seperation(im, type=1) for im in image_file]
    })
    return df


if __name__ == "__main__":
    df = get_dataset("P:\\LPW\\LPW\\")
    analysis_dict = dict()
    for i, row in df.iterrows():
        print("\n" + row["category"])
        analysis = analysis_base(row)
        pd.DataFrame([analysis]).transpose().to_excel(resultPath + "\\" + row["category"] + "_" + start_t + ".xlsx")
        analysis_dict[row["category"]] = analysis
    pd.DataFrame(analysis_dict).to_excel(resultPath + "\\ALL" + "_" + start_t + ".xlsx")
    average_all = get_total_analysis_results(analysis_dict)
    average_all["traditional_accuracy"] = None
    pd.DataFrame(average_all).to_excel(resultPath + "\\ALL_AVG" + "_" + start_t + ".xlsx")

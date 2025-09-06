from dikablish_trad_conf import *
import numpy as np
import torch
import glob
import math
import matplotlib.pyplot as plt
import os
import cv2
import skimage.exposure as exposure

device = "cuda:0"
resolution = 320 * 240

def perform_detection(image):
    global fully_model
    with torch.no_grad(): map = fully_model(image.to(device))
    return map

def init_model():
    global patch_model, fully_model
    fully_model = torch.load(model_file).to(device)
    tns = torch.randn((1, 1, 320, 240)).to(device)
    for _ in range(3):
        with torch.no_grad(): _ = fully_model(tns)

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

def ss_nick_dikablish(dsi):
    nick = dsi.split("\\")[-1].split("_")
    nick = nick[0] + "_" + nick[1] + "_" + nick[2] + "_"
    return nick



dikablish_ss = get_dikablish_ss()
acc, err, list_x, list_y, list_h, list_w, list_a = 0, 0, [], [], [], [], []
ds_count = 0
fully_model = None
previous_inference, previous_mask = None, None
#model_file = "C:\\Users\\Sheriffnnr\\Desktop\\Fixtare Pupil Detection\\UnetPP_S\\UNetPP_FULL_FULL_FPD_25_l0.019_d0.962_vd0.894.pt"
#model_file = "C:\\Users\\Sheriffnnr\\Desktop\\Fixtare Pupil Detection\\UnetPP_S\\UNetPP-s_FULL_FPD_24_l0.019_d0.962_vd0.888.pt"
model_file = "X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt"
init_model()


for ss in dikablish_ss:
    new_ds = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format(ss)), key=list_sort_key_dikablish)
    nick = ss_nick_dikablish(new_ds[0])

    for dsi in new_ds:
        image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
        mask_ellipse = get_ellipse_parameters(mask)

        temp = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        temp = torch.tensor(temp.astype(np.float32) / 255.).unsqueeze(0)
        map = perform_detection(temp)
        map = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
        #map[np.nonzero(map < 128)] = 0.0
        #map[np.nonzero(map >= 128)] = 255

        ellipse = get_ellipse_parameters(map)
        if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): mask_ellipse = previous_mask
        control = False
        if ((ellipse[0][0] == 0 and ellipse[0][1] == 0)):
            ellipse, control = previous_inference, True


        px_err = math.dist(ellipse[0], mask_ellipse[0])
        acc = acc + (1 if px_err <= 5 else 0)
        err = err + px_err
        list_h.append(abs(mask_ellipse[1][0] - ellipse[1][0]))
        list_w.append(abs(mask_ellipse[1][1] - ellipse[1][1]))
        list_a.append(abs(mask_ellipse[2] - ellipse[2]))
        list_x.append(abs(ellipse[0][0] - ellipse[0][0]))
        list_y.append(abs(ellipse[0][1] - ellipse[0][1]))

        if px_err > 5:
            print("[" + str(ds_count) + "]", dsi, px_err, "   ", err)
            if False and ds_count > 190:
                print("pred", ellipse)
                print("mask", mask_ellipse)

        previous_mask = mask_ellipse
        previous_inference = ellipse

        ds_count += 1


list_x, list_y, list_w, list_h, list_a = np.asarray(list_x), np.asarray(list_y), np.asarray(list_w), np.asarray(list_h), np.asarray(list_a)
print("x", np.mean(list_x))
print("y", np.mean(list_y))
print("w", np.mean(list_w))
print("h", np.mean(list_h))
print("a", np.mean(list_a))
print("\n\nds_count: {0}\n"
      "acc: {1}\n"
      "norm acc: {2}\n"
      "err: {3}\n"
      "norm err: {4}\n".format(
    ds_count,
    acc, acc / ds_count, err, err / ds_count
))

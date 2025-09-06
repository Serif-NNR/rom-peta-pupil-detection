import glob, cv2, math
import numpy as np
import copy
import sys
import cv
import ellipse
import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import segmentation_models_pytorch as smp
from LEyesEvaluate.LEyes_evaluate2 import do_hourglass_prediction, get_leyes_model, do_leyes_XX
import matplotlib.pyplot as plt


class PupilNet(nn.Module):
    def __init__(self):
        super(PupilNet, self).__init__()
        self.backbone = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )

    def forward(self, inputs):
        return self.backbone(inputs)

    def predict(self, inputs, sfac=None, offsets=None):
        with torch.no_grad():
            pred = self.forward(inputs)
        pred = nn.Sigmoid()(pred).detach().cpu().squeeze()
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        mask = pred > 0.99

        pupils = []
        for i, m in enumerate(mask):
            pup = cv.detect_pupil_from_thresholded(m.numpy().astype('uint8') * 255, symmetry_tresh=0.3,
                                                   kernel=cv.kernel_pup2)
            pupil = {'centroid': (pup['ellipse'][0]),
                     'axis_major_radius': pup['ellipse'][1][0] / 2,
                     'axis_minor_radius': pup['ellipse'][1][1] / 2,
                     'orientation': pup['ellipse'][2]
                     }
            max_rad = max(pupil['axis_major_radius'], pupil['axis_minor_radius'])
            pupil['too_close_edge'] = pupil['centroid'][0] < max_rad or pupil['centroid'][0] > m.shape[1] or \
                                      pupil['centroid'][1] < max_rad or pupil['centroid'][1] > m.shape[0]

            if (not np.isnan(pupil['centroid'][0])) and (sfac is not None):
                el = ellipse.my_ellipse((*(pupil['centroid']), pupil['axis_major_radius'], pupil['axis_minor_radius'],
                                         pupil['orientation'] / 180 * np.pi))
                tform = ellipse.scale_2d(sfac, sfac)
                nelpar = el.transform(tform)[0][:-1]
                pupil['oripupil'] = copy.deepcopy(pupil)
                pupil['centroid'] = (nelpar[0], nelpar[1])
                pupil['axis_major_radius'] = nelpar[2]
                pupil['axis_minor_radius'] = nelpar[3]
                pupil['orientation'] = nelpar[4] / np.pi * 180

            if (not np.isnan(pupil['centroid'][0])) and (offsets[i] is not None):
                if sfac is None:
                    pupil['oripupil'] = copy.deepcopy(pupil)
                pupil['centroid'] = tuple(x + y for x, y in zip(pupil['centroid'], offsets[i].numpy().flatten()))

            pupil["mask"] = m
            pupil["pmap"] = pred[i]
            pupils.append(pupil)

        return pupils


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


def read_file_as_dictionary(path):
    d = dict()
    f = open(path, "r")
    line = f.readlines()
    for l in line:
        s = l.split(";")
        d[s[0]] = s
    return d


def AnalyzePureAndHaar():
    dictHaar = read_file_as_dictionary("C:\\Users\\Sheriffnnr\\Desktop\\PD_AfterReview\\Banchmark\\fileHaar2.txt")
    dictPure = read_file_as_dictionary("C:\\Users\\Sheriffnnr\\Desktop\\PD_AfterReview\\Banchmark\\filePure2.txt")
    ds_count, acc_haar, acc_pure, err_haar, err_pure = 0, 0, 0, 0, 0
    list_h, list_w, list_a = [], [], []
    for dsi in ds:
        image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
        mask_ellipse = get_ellipse_parameters(mask)

        probHaar = dictHaar[dsi]
        probPure = dictPure[dsi]

        pxHaar = math.dist((float(probHaar[1]), float(probHaar[2])), mask_ellipse[0])
        pxPure = math.dist((float(probPure[1]), float(probPure[2])), mask_ellipse[0])

        err_haar += pxHaar
        err_pure += pxPure

        acc_haar = acc_haar + (1 if pxHaar <= 5 else 0)
        acc_pure = acc_pure + (1 if pxPure <= 5 else 0)

        list_h.append(abs(mask_ellipse[1][0] - float(probHaar[3])))
        list_w.append(abs(mask_ellipse[1][1] - float(probHaar[4])))
        list_a.append(abs(mask_ellipse[2] - float(probHaar[5])))

        if pxPure > 5 or pxHaar > 5:
            print("[{4}] PURE-> {0}: {1}\n[{5}] HAAR-> {2}: {3}".format(probPure, pxPure, probHaar, pxHaar, ds_count,
                                                                        ds_count))
        ds_count += 1

    list_w, list_h, list_a = np.asarray(list_w), np.asarray(list_h), np.asarray(list_a)
    print(np.mean(list_w))
    print(np.mean(list_h))
    print(np.mean(list_a))

    print("\n\nds_count: {0}\n"
          "acc_haar: {1}\n"
          "norm acc_haar: {2}\n"
          "err_haar: {3}\n"
          "norm err_haar: {4}\n"
          "acc_pure: {5}\n"
          "norm acc_pure: {6}\n"
          "err_pure: {7}\n"
          "norm err_pure: {8}\n".format(
        ds_count,
        acc_haar, acc_haar / ds_count, err_haar, err_haar / ds_count,
        acc_pure, acc_pure / ds_count, err_pure, err_pure / ds_count,
    ))


def AnalyzeLEye():
    dictPure = read_file_as_dictionary("C:\\Users\\Sheriffnnr\\Desktop\\PD_AfterReview\\Banchmark\\filePure.txt")
    model = get_leyes_model()
    ds_count, acc, err = 0, 0, 0
    list_h, list_w, list_a = [], [], []
    previous_mask, previous_inference = None, None
    dikablish_ss = get_dikablish_ss()
    repredicted_count = 0
    for ss in dikablish_ss:
        new_ds = sorted(glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\{0}_*_I.png'.format(ss)),
                        key=list_sort_key_dikablish)
        for dsi in new_ds:
            image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
            mask_ellipse = get_ellipse_parameters(mask)

            center, map, ox, oy, output, ellipse, repredicted = do_leyes_XX(model, "cuda:0", image, dictPure[dsi], 0.7, 128, 1)
            repredicted_count += repredicted
            # mask = mask[oy:oy + 128, ox:ox + 128]
            # mask_ellipse = get_ellipse_parameters(mask)
            if False:
                out_cut = np.copy(p[0]["pmap"].detach().cpu().numpy())
                plt.imshow(out_cut)
                plt.show()
                print("stop")

            '''if center[0] == 0 and center[1] == 0 and (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): continue
            if center[0] == 0 and center[1] == 0:
                continue
                #center = previous_inference
            if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0) and previous_mask != None: mask_ellipse = previous_mask
            if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): continue'''
            if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): mask_ellipse = previous_mask
            if ((ellipse[0][0] == 0 and ellipse[0][1] == 0) or ellipse[1][0] < 5 or ellipse[1][1] < 5): center = previous_inference

            px_err = math.dist(center, mask_ellipse[0])
            acc = acc + (1 if px_err <= 5 else 0)
            err = err + px_err
            list_h.append(abs(mask_ellipse[1][0] - ellipse[1][0]))
            list_w.append(abs(mask_ellipse[1][1] - ellipse[1][1]))
            list_a.append(abs(mask_ellipse[2] - ellipse[2]))
            # print(dsi)
            if px_err > 5:
                print("[" + str(ds_count) + "]", dsi, px_err, "   ", err, "\tRP:", repredicted_count)
                if False:
                    print(mask_ellipse)
                    plt.figure()
                    f, axarr = plt.subplots(3, 1)
                    pred = np.zeros((240, 320))
                    pred[oy:oy + 128, ox:ox + 128] = map
                    axarr[0].imshow(pred)
                    axarr[1].imshow(mask)
                    axarr[2].imshow(output)
                    plt.show()

            ds_count += 1
            previous_mask = mask_ellipse
            previous_inference = center
    list_w, list_h, list_a = np.asarray(list_w), np.asarray(list_h), np.asarray(list_a)
    print(np.mean(list_w))
    print(np.mean(list_h))
    print(np.mean(list_a))
    print("\n\nds_count: {0}\n"
          "acc: {1}\n"
          "norm acc: {2}\n"
          "err: {3}\n"
          "norm err: {4}\n".format(
        ds_count,
        acc, acc / ds_count, err, err / ds_count
    ))


if __name__ == "__main__":
    AnalyzeLEye()
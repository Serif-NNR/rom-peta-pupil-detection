import glob
import os
import time
import cv2
import torch
from fixein_traditional_method import *
from excuse_else_category import excuse_else_finetuning
from pupil_net_category import pupilnet_finetuning
from segmodel import UNet, UNetFirst
from train_and_metric import device
import matplotlib.pyplot as plt
import pandas as pd

image_files = glob.glob('X:\\Pupil Dataset\\ExCuSe\\*\\*.png')
image_files += glob.glob('X:\\Pupil Dataset\\Else\\*\\*.png')
cp_files = glob.glob('X:\\Pupil Dataset\\ExCuSe\\*.txt')

def get_category(image_path):
  return image_path.split('\\')[3]

def get_imagecount(image_path):
  return image_path.split('\\')[4].split('.')[0]

df = pd.DataFrame({"image_path": image_files,
                  "category": [get_category(x) for x in image_files],
                   "image_count": [get_imagecount(x) for x in image_files]
                   })


def intTryParse(value):
    try:
        a = int(value)
        return True
    except ValueError:
        return False


model = UNetFirst(1, 1).to(device)
modelPath = "167619080976241\\FPD_19_l0.148_d0.899_vd0.934.pt"
model = torch.load("ModelParameters\\" + modelPath)
resultPath = "RawResultAddUp2\\" + modelPath.split('\\')[0]+"_" + str(time.time()).replace('.', '')
os.mkdir(resultPath)
i = 0
for category in df["category"].unique():
    i += 1
    #if i == 1: continue
    print(category)
    count = 0
    pd, th = excuse_else_finetuning[category][0], excuse_else_finetuning[category][1]
    resultVideo = cv2.VideoWriter(resultPath + "\\" + category+".avi", cv2.VideoWriter_fourcc(*'MJPG'), 20,(346, 288-28))
    resultVideoCropped = cv2.VideoWriter(resultPath + "\\CROPPED_" + category+".avi",cv2.VideoWriter_fourcc(*'MJPG'), 20,(346, 288-28))
    predDurNormal = []
    predDurCropped = []


    for i, dsrow in df[(df["category"] == category)].iterrows():
        image1 = cv2.imread(dsrow["image_path"])
        mask1 = cv2.imread(dsrow["image_path"])
        xcrop = int(image1.shape[1] * 0.05)
        ycrop = int(image1.shape[0] * 0.05)
        yshape = image1.shape[0]
        xshape = image1.shape[1]
        image = image1[ycrop:(yshape - ycrop), 0 + xcrop:xshape - xcrop, 0]
        mask = mask1[ycrop:(yshape - ycrop), 0 + xcrop:xshape - xcrop, 0]


        # Traditional
        try:
            minmax = procedure_min_max_simple(image)
            smim = filter_savitzky_golay(minmax)
            start_x, end_x, start_y, end_y, thx, thy = find_pupil_area(smim[0:2], pd, th)

        except: continue
        if start_x - end_x < 30:
            start_x -= 15
            end_x += 15
            if start_x < 0: start_x = 0
            if end_x > 384-40: end_x = 384-40
        if start_y - end_y < 30:
            start_y -= 15
            end_y += 15
            if start_y < 0: start_y = 0
            if end_y > 288-30: end_y = 288-30
        start_point = (start_x, start_y)
        end_point = (end_x, end_y)
        result = draw_rect_on_image(start_point, end_point,
                                    smim, image, thx, thy)
        resmask = draw_rect_on_image(start_point, end_point,
                                     smim, mask, thx, thy)
        acc = calculate_mask_acc(start_point, end_point, mask)
        cropped_image = image[start_y:end_y, start_x:end_x]

        # Learning
        startingTime = time.time()
        predCropped = np.expand_dims(cropped_image, axis=-1).transpose((2, 0, 1))
        predCropped = torch.tensor(predCropped.astype(np.float32) / 255.).unsqueeze(0)
        try:
            predCropped = model(predCropped.to(device))
        except:
            continue
        raw_predicted_image_cropped = (predCropped.detach().cpu().numpy()[0, 0, :, :] * 200).astype(np.uint8)

        #img_np = np.where(raw_predicted_image_cropped > (np.min(raw_predicted_image_cropped)+np.max(raw_predicted_image_cropped))/2, 255, 0).astype(np.int8)
        #gray = cv2.cvtColor(raw_predicted_image_cropped, cv2.COLOR_BGR2GRAY)
        thxxx, threshed = cv2.threshold(raw_predicted_image_cropped,
                                     20,
                                     255, cv2.THRESH_BINARY)
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
            raw_predicted_image_cropped = cv2.circle(raw_predicted_image_cropped, (int(ellipse[0][0]), int(ellipse[0][1])), 4, (100, 100, 100), -1)
            #plt.imshow(raw_predicted_image_cropped)
            #plt.show()

        predDurCropped.append(time.time()-startingTime)

        startingTime = time.time()
        pred = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        pred = torch.tensor(pred.astype(np.float32) / 255.).unsqueeze(0)
        pred = model(pred.to(device))
        raw_predicted_image = (pred.detach().cpu().numpy()[0, 0, :, :] * 256).astype(int)
        predDurNormal.append(time.time()-startingTime)

        ## Visualization
        resultCropped = result.copy()
        #resultCropped = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        result[:, :, 2] = 0
        resultCropped[start_y:end_y, start_x:end_x, 2] = raw_predicted_image_cropped
        result[:, :, 2] = raw_predicted_image

        #plt.imshow(resultCropped)
        #plt.show()
        resultVideo.write(result)
        resultVideoCropped.write(resultCropped)

        count += 1
        # if count == 50: success = False
    print("Video   : {0}".format(category))
    print("Normal  : {0}".format(np.sum(predDurNormal) / count))
    print("Cropped : {0}\n\n".format(np.sum(predDurCropped) / count))
    resultVideo.release()
    resultVideoCropped.release()

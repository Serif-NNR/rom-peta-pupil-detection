import glob
import os
import time
import cv2
import torch
from fixein_traditional_method import *
from lpw_category import finetuning
from segmodel import UNet, UNetFirst, MyNet, MyNet2, DynamicNet
from train_and_metric import device
import matplotlib.pyplot as plt

image_file_full = glob.glob('Fixein LPW/*_*_*/*.avi')

##torch.cuda.synchronize()

def get_masked_video(image_path):
    fileName = image_path.split('\\')[2].split('.')[0].split('_')
    return "Fixein LPW\\Pupils in the wild improved\\folder-{0}_file-{1}_pupil.mp4".format(fileName[0], fileName[1])

#1679427061031889
model = UNet(1, 1).to(device)
model.zero_grad()
#modelPath = "167619080976241\\FPD_19_l0.148_d0.899_vd0.934.pt" # UNetFirst
#modelPath = "1679427061031889\\FPD_16_l0.129_d0.889_vd0.942.pt" # MyNet
modelPath = "1679522336797663\\FPD_92_l0.150_d0.871_vd0.919.pt" # Dynamic
modelPath = "1679678154396987\\FPD_18_l0.152_d0.894_vd0.956.pt" # UNet FULL

modelPath = "16797869820526958\\FPD_22_l0.068_d0.946_vd0.838.pt" # UNet FULL NEW DATASET
modelPath = "16804570461139164_SegNet_VGG16\\FPD_34_l0.066_d0.945_vd0.776.pt" # UNet FULL ORI DATASET
model = torch.load("ModelParameters\\" + modelPath)
resultPath = "RawResultAddUp2\\" + modelPath.split('\\')[0]+"_" + str(time.time()).replace('.', '')
os.mkdir(resultPath)

for video in image_file_full:
    image_video = cv2.VideoCapture(video)
    video_mask = get_masked_video(video)
    mask_video = cv2.VideoCapture(video_mask)

    success, image = image_video.read()
    _, mask = mask_video.read()
    count = 0
    pd, th = finetuning[video.split('\\')[2]][0], finetuning[video.split('\\')[2]][1]

    resultVideo = cv2.VideoWriter(resultPath+"\\"+video.split('\\')[2], cv2.VideoWriter_fourcc(*'MJPG'), 60, (320, 240))
    resultVideoCropped = cv2.VideoWriter(resultPath + "\\CROPPED_" + video.split('\\')[2], cv2.VideoWriter_fourcc(*'MJPG'), 60,
                                  (320, 240))
    predDurNormal = []
    predDurCropped = []
    while success:
        time.sleep(0.001)
        image, mask = cv2.resize(image, (320, 240), cv2.INTER_AREA), cv2.resize(mask, (320, 240), cv2.INTER_AREA)
        image = image[:, :, 0]
        mask = mask[:, :, 0]

        # Traditional
        minmax = procedure_min_max_simple(image)
        smim = filter_savitzky_golay(minmax)
        start_x, end_x, start_y, end_y, thx, thy = find_pupil_area(smim[0:2], pd, th)
        if start_x - end_x < 60:
            start_x -= 30
            end_x += 30
            if start_x < 0: start_x = 0
            if end_x > 320: end_x = 320
        if start_y - end_y < 30:
            start_y -= 15
            end_y += 15
            if start_y < 0: start_y = 0
            if end_y > 240: end_y = 240
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
        time.sleep(0.001)
        predCropped = model(predCropped.to(device))

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
        time.sleep(0.001)
        raw_predicted_image = (pred.detach().cpu().numpy()[0, 0, :, :] * 256).astype(int)
        predDurNormal.append(time.time()-startingTime)

        ## Visualization
        resultCropped = result.copy()
        #resultCropped = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        result[:, :, 2] = 0
        resultCropped[start_y:end_y, start_x:end_x, 2] = raw_predicted_image_cropped
        result[:, :, 2] = raw_predicted_image

        #plt.imshow(result)
        #plt.show()
        resultVideo.write(result)
        resultVideoCropped.write(resultCropped)

        count += 1
        success, image = image_video.read()
        _, mask = mask_video.read()
        # if count == 50: success = False
    print("Video   : {0}".format(video))
    print("Normal  : {0}".format(np.sum(predDurNormal) / count))
    print("Cropped : {0}\n\n".format(np.sum(predDurCropped) / count))
    resultVideo.release()
    resultVideoCropped.release()

import glob, cv2
import numpy as np
import matplotlib.pyplot as plt


def get_annot_dict(video_path):
    annot_dict = dict()
    for path in video_path:
        annot_path = "X:\\Pupil Dataset\\Dikablis\\" + path.split("\\")[-1] + "pupil_eli.txt"
        video_name = path.split("\\")[-1].split(".")[0]
        file = open(annot_path, "r")
        context = file.readlines()
        for c in context:
            part = c.split(";")
            if part[0] != "FRAME" and \
                        "DikablisT_1_9" != video_name  and "DikablisT_20_12" != video_name \
                    and "DikablisT_22_7" != video_name and "DikablisT_22_8" != video_name  \
                    and "DikablisT_22_9" != video_name and "DikablisT_23_3" != video_name  \
                    and "DikablisT_23_5" != video_name and "DikablisT_24_11" != video_name \
                    and "DikablisT_24_5" != video_name and "DikablisT_24_8" != video_name \
                    and "DikablisSA_5_2" != video_name and "DikablisSS_17_2" != video_name:
                part[1] = float(part[1])
                part[2] = float(part[2])
                part[3] = float(part[3])
                part[4] = float(part[4])
                part[5] = float(part[5])
                annot_dict[video_name + "_" + str(int(part[0])-1)] = part[0:-1]
    return annot_dict

def get_mask(annot_code):
    ellp = annot_dict[annot_code]
    if ellp[1] != -1:
        mask = cv2.ellipse(np.zeros((288, 384)), (int(ellp[2]), int(ellp[3])),
                           (int(ellp[4] * .5), int(ellp[5] * .5)), int(ellp[1]), 0, 360, 255, -1)
        #mask = cv2.resize(mask, (320, 240))
    else:
        mask = np.zeros((288, 384))
    if False:
        plt.imshow(image + mask)
        plt.show()
        plt.imshow(mask)
        plt.show()
    return mask


video_paths = glob.glob('X:\\Pupil Dataset\\Dikablis\\*.mp4')
video_counter, total_video_count = 1, len(video_paths)
annot_dict = get_annot_dict(video_paths)

for path in video_paths:
    video = cv2.VideoCapture(path)
    video_name = path.split('\\')[-1].split('.')[0]
    print("[" + str(video_counter) + "/" + str(total_video_count) + "]", video_name)
    success, image = video.read()
    count = 0
    while success:
        try:
            image, mask = image[14:274, 19:365, 0], get_mask(video_name + "_" + str(count))[14:274, 19:365]
        except:
            break
        image, mask = cv2.resize(image, (320, 240), cv2.INTER_AREA), cv2.resize(mask, (320, 240), cv2.INTER_AREA)
        if False:
            f, axarr = plt.subplots(2, 3)
            axarr[0, 0].imshow(image)
            axarr[0, 1].imshow(mask)
            plt.show()
        cv2.imwrite("X:\\Pupil Dataset\\Dikablis_Full\\" + video_name + "_" + str(count) + "_I.png", image)
        cv2.imwrite("X:\\Pupil Dataset\\Dikablis_Full\\" + video_name + "_" + str(count) + "_M.png", mask)
        count += 1
        success, image = video.read()
    video_counter += 1

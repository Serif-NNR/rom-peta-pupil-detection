import glob, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
from train_and_metric import dice_coef_metric
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
                    and "DikablisT_24_5" != video_name and "DikablisT_24_8" != video_name:
                part[1] = float(part[1])
                part[2] = float(part[2])
                part[3] = float(part[3])
                part[4] = float(part[4])
                part[5] = float(part[5])
                annot_dict[video_name + "_" + str(int(part[0])-1)] = part[0:-1]
    return annot_dict

def get_image_and_map(path, annot_code):
    ellp = annot_dict[annot_code]
    image = cv2.imread(path)[:, :, 0]
    if ellp[1] != -1:
        mask = cv2.ellipse(np.zeros((288, 384)), (int(ellp[2]), int(ellp[3])),
                           (int(ellp[4] * .5), int(ellp[5] * .5)), int(ellp[1]), 0, 360, 255, -1)
        mask = cv2.resize(mask, (320, 240))
    else:
        mask = np.zeros((240, 320))
    if image_count % 10_000 == 1 and False:
        plt.imshow(image + mask)
        plt.show()
        plt.imshow(mask)
        plt.show()
    return image, mask



video_paths = glob.glob('X:\\Pupil Dataset\\Dikablis\\*.mp4')
image_paths = glob.glob('X:\\Pupil Dataset\\Dikablis_Full\\*.png')
annot_dict = get_annot_dict(video_paths)

#model_param_dir = "ModelParameters/1688857672_UNet_CROP_FULL9279747/FPD_17_l0.019_d0.966_vd0.758.pt"
#model_param_dir = "X:\\Fixein Pupil Detection\\ModelParameters\\UNET MODELS\\UNet_R41_R41_FPD_17_l0.032_d0.950_vd0.842.pt"
model_param_dir = "X:\\Fixein Pupil Detection\\ModelParameters\\UNET MODELS\\UNet_AUG_FULL_FPD_25_l0.031_d0.951_vd0.890.pt"
#model_param_dir = "X:\\Fixein Pupil Detection\\ModelParameters\\1689528522_UNet_BAUG_FULL1728463\\FPD_19_l0.020_d0.964_vd0.882.pt"
#model_param_dir = "X:\\Fixein Pupil Detection\\ModelParameters\\1689421744_UNet_FULL_FULL6306918\\FPD_4_l0.018_d0.967_vd0.897.pt"
model = torch.load(model_param_dir).to(device)


image_count, image_total = 1, len(image_paths)


list_dice = list()


for path in image_paths:
        print("["+str(image_count)+"/"+str(image_total)+"]", path)
        annot_code = path.split("\\")[-1].split(".")[0]
        if annot_code in annot_dict:
            image, mask = get_image_and_map(path, annot_code)

            image = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
            image = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0)
            mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
            mask = torch.tensor(mask.astype(np.float32) / 255.).unsqueeze(0)

            with torch.no_grad():
                output = model(torch.tensor(image).to(device))
            out_cut = np.copy(output.detach().cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            mask = mask.detach().cpu().numpy()

            train_dice = dice_coef_metric(out_cut, mask)
            list_dice.append(train_dice)

        image_count += 1

print("DICE: ", np.asarray(list_dice).mean())


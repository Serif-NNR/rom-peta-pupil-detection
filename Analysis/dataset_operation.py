import glob
import os
import random
from fixein_traditional_method import *
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch as torch
from lpw_category import finetuning


class ConvertLPW_VideoToImage(object):

    def BatchlessCroppedImage(self):
        random.seed(7)
        dirName = "FixeinLPWimageCroppedBatchAug"
        controlAug = True

        os.mkdir(dirName)
        os.mkdir(dirName + "/0_0")
        os.mkdir(dirName + "/0_1")
        os.mkdir(dirName + "/0_2")
        os.mkdir(dirName + "/1_0")
        os.mkdir(dirName + "/1_1")
        os.mkdir(dirName + "/1_2")
        os.mkdir(dirName + "/2_0")
        os.mkdir(dirName + "/2_1")
        os.mkdir(dirName + "/2_2")

        image_file_full = glob.glob('Fixein LPW/*_*_*/*.avi')

        def get_masked_video(image_path):
            fileName = image_path.split('\\')[2].split('.')[0].split('_')
            return "Fixein LPW\\Pupils in the wild improved\\folder-{0}_file-{1}_pupil.mp4".format(fileName[0],
                                                                                                   fileName[1])

        onehot = {"Clear": "0", "Clean": "0", "Blurry": "1", "Cloudy": "2", "None": "0", "Small": "1", "Giant": "2"}
        for video in image_file_full:
            image_video = cv2.VideoCapture(video)
            video_mask = get_masked_video(video)
            mask_video = cv2.VideoCapture(video_mask)

            success, image = image_video.read()
            success, mask = mask_video.read()
            count = 0

            pd, th = finetuning[video.split('\\')[2]][0], finetuning[video.split('\\')[2]][1]

            while success:

                image = cv2.resize(image, (320, 240), cv2.INTER_AREA)
                mask = cv2.resize(mask, (320, 240), cv2.INTER_AREA)
                category = "{0}_{1}".format(onehot[video.split('\\')[1].split('_')[1]],
                                            onehot[video.split('\\')[1].split('_')[2]])
                image = image[:, :, 0]
                for i in range(6 if controlAug else 1):
                    if count % 10 == 0:
                        minmax = procedure_min_max_simple(image)
                        smim = filter_savitzky_golay(minmax)
                        start_x, end_x, start_y, end_y, thx, thy = find_pupil_area(smim[0:2], pd, th)
                        if start_x - end_x < 20:
                            start_x -= 10
                            end_x += 10
                            if start_x < 0: start_x = 0
                            if end_x > 320: end_x = 320
                        if start_y - end_y < 20:
                            start_y -= 10
                            end_y += 10
                            if start_y < 0: start_y = 0
                            if end_y > 240: end_y = 240
                    resim = image[start_y:end_y, start_x:end_x]
                    resmask = mask[start_y:end_y, start_x:end_x]

                    if controlAug:
                        aug = A.Compose([
                            A.VerticalFlip(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.1),
                            # A.GridDistortion(p=0.1),
                            A.Blur(p=0.2),
                            A.RandomBrightnessContrast(p=0.3),
                            A.GaussNoise(p=0.1),
                            A.Sharpen(p=0.1),
                            # A.Perspective(p=0.1)
                        ])
                        augmented = aug(image=resim, mask=resmask)
                        resim = augmented["image"]  # image[y:y+120, x:x+105]
                        resmask = augmented["mask"]  # mask[y:y+120, x:x+105]

                    cv2.imwrite(dirName + "/" + category + "/I_" + video.split('\\')[2] + "_" + str(count) + "_" + str(
                        i) + ".png", resim)
                    cv2.imwrite(dirName + "/" + category + "/M_" + video.split('\\')[2] + "_" + str(count) + "_" + str(
                        i) + ".png", resmask)
                count += 1
                success, image = image_video.read()
                success, mask = mask_video.read()


    def CroppedImage(self):
        random.seed(7)
        dirName = "FixeinLPWimageAug"

        os.mkdir(dirName)
        os.mkdir(dirName+"/0_0")
        os.mkdir(dirName+"/0_1")
        os.mkdir(dirName+"/0_2")
        os.mkdir(dirName+"/1_0")
        os.mkdir(dirName+"/1_1")
        os.mkdir(dirName+"/1_2")
        os.mkdir(dirName+"/2_0")
        os.mkdir(dirName+"/2_1")
        os.mkdir(dirName+"/2_2")

        image_file_full = glob.glob('Fixein LPW/*_*_*/*.avi')

        def get_masked_video(image_path):
            fileName = image_path.split('\\')[2].split('.')[0].split('_')
            return "Fixein LPW\\Pupils in the wild improved\\folder-{0}_file-{1}_pupil.mp4".format(fileName[0],
                                                                                                fileName[1])

        onehot = {"Clear": "0", "Clean": "0", "Blurry": "1", "Cloudy": "2", "None": "0", "Small": "1", "Giant": "2"}
        for video in image_file_full:
            image_video = cv2.VideoCapture(video)
            video_mask = get_masked_video(video)
            mask_video = cv2.VideoCapture(video_mask)

            success, image = image_video.read()
            success, mask = mask_video.read()
            count = 0

            while success:
                image = cv2.resize(image, (320, 240), cv2.INTER_AREA)
                mask = cv2.resize(mask, (320, 240), cv2.INTER_AREA)
                category = "{0}_{1}".format(onehot[video.split('\\')[1].split('_')[1]],
                                            onehot[video.split('\\')[1].split('_')[2]])
                for i in range(6):
                    x = random.randint(0, 214)
                    y = random.randint(0, 119)

                    aug = A.Compose([
                        A.VerticalFlip(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        #A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.1),
                        #A.GridDistortion(p=0.1),
                        A.Blur(p=0.2),
                        A.RandomBrightnessContrast(p=0.3),
                        A.GaussNoise(p=0.1),
                        A.Sharpen(p=0.1),
                        #A.Perspective(p=0.1)
                    ])
                    #augmented = aug(image=image[y:y+120, x:x+105], mask=mask[y:y+120, x:x+105])
                    augmented = aug(image=image, mask=mask)
                    augimage = augmented["image"] #image[y:y+120, x:x+105]
                    augmask = augmented["mask"]   #mask[y:y+120, x:x+105]

                    cv2.imwrite(dirName+"/" + category + "/I_" + video.split('\\')[2] + "_" + str(count) + "_" + str(i) + ".png", augimage)
                    cv2.imwrite(dirName+"/" + category + "/M_" + video.split('\\')[2] + "_" + str(count) + "_" + str(i) + ".png", augmask)
                count += 1
                success, image = image_video.read()
                success, mask = mask_video.read()
                # if count == 50: success = False


    def StandartImage(self):
        dirName = "FixeinLPWimage"
        os.mkdir(dirName)
        os.mkdir(dirName+"/0_0")
        os.mkdir(dirName+"/0_1")
        os.mkdir(dirName+"/0_2")
        os.mkdir(dirName+"/1_0")
        os.mkdir(dirName+"/1_1")
        os.mkdir(dirName+"/1_2")
        os.mkdir(dirName+"/2_0")
        os.mkdir(dirName+"/2_1")
        os.mkdir(dirName+"/2_2")

        image_file_full = glob.glob('Fixein LPW/*_*_*/*.avi')

        def get_masked_video(image_path):
            fileName = image_path.split('\\')[2].split('.')[0].split('_')
            return "Fixein LPW\\Pupils in the wild improved\\folder-{0}_file-{1}_pupil.mp4".format(fileName[0],
                                                                                                fileName[1])

        onehot = {"Clear": "0", "Clean": "0", "Blurry": "1", "Cloudy": "2", "None": "0", "Small": "1", "Giant": "2"}
        for video in image_file_full:
            image_video = cv2.VideoCapture(video)
            video_mask = get_masked_video(video)
            mask_video = cv2.VideoCapture(video_mask)

            success, image = image_video.read()
            success, mask = mask_video.read()
            count = 0
            while success:
                image = cv2.resize(image, (320, 240), cv2.INTER_AREA)
                mask = cv2.resize(mask, (320, 240), cv2.INTER_AREA)
                category = "{0}_{1}".format(onehot[video.split('\\')[1].split('_')[1]],
                                            onehot[video.split('\\')[1].split('_')[2]])
                cv2.imwrite(dirName+"/" + category + "/I_" + video.split('\\')[2] + "_" + str(count) + ".png", image)
                cv2.imwrite(dirName+"/" + category + "/M_" + video.split('\\')[2] + "_" + str(count) + ".png", mask)
                count += 1
                success, image = image_video.read()
                success, mask = mask_video.read()
                # if count == 50: success = False

if __name__=="__main__":
    #ConvertLPW_VideoToImage()
    ConvertLPW_VideoToImage().BatchlessCroppedImage()
    pass

class PupilDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 0], 0)
        image = np.array(image) / 255.
        mask = cv2.imread(self.df.iloc[idx, 1], 0)
        mask = np.array(mask) / 255.

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        # image = image.transpose((2,0,1))
        image = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        image = torch.from_numpy(image).type(torch.float32)
        # image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image)
        mask = np.expand_dims(mask, axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask).type(torch.float32)

        return image, mask, self.df.iloc[idx, 0]

def my_collate(batch):
    # data = [item[0].numpy() for item in batch]
    # target = [item[1].numpy() for item in batch]

    # target = torch.LongTensor(target)

    # target = target[0]
    # data = data[0]
    # data = torch.from_numpy(data)
    # target = target.to(torch.float32)

    # data = data.unsqueeze(0)
    # target = target.unsqueeze(0)
    # return [data, target]
    d = batch[0][0].unsqueeze(0)
    t = batch[0][1].unsqueeze(0)
    return [d, t, batch[0][2]]



class DatasetAttributes(object):
    #dirName = "FixeinLPWimageCroped"
    #dirName = "FixeinLPWimage"
    #dirName = "FixeinLPWimageCropedAug"
    #dirName = "FixeinLPWimageCropedAugSmooth"
    df_full = None

    train_df_full = None
    test_df_full = None
    val_df_full = None

    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    def __init__(self, dirName, new_ds=False, complex_ds = False):
        self.dirName = dirName
        self.new_ds = new_ds
        self.complex_ds = complex_ds
        self.lpw_variation = [
            "P:\\LPW\\LPW_FULL\\",
            "P:\\LPW\\LPW_CROP_x2\\",
            "P:\\LPW\\LPW_CROP_x4\\",
            "P:\\LPW\\LPW_CROP\\",
            "P:\\LPW\\LPW_CROP_x9\\",
            "P:\\LPW\\LPW_CROP_x16\\",
            "P:\\LPW\\LPW_CROP_x25\\",
        ]

    def Shortly(self, RATE=1, name=""):
        if self.complex_ds: self.GetDatasetDataframeComplex(RATE=RATE)
        elif self.new_ds: self.GetDatasetDataframeNew(RATE=RATE)
        else: self.GetDatasetDataframe(RATE=RATE)
        self.GetSubsets()
        self.PrepareDataloader()
        self.name = name

    def GetDatasetDataframeComplex(self, RATE=1):
        mask_files_full, image_files_full = [], []
        for var in self.lpw_variation:
            list_new = glob.glob(var + '*_M.png')
            mask_files_full += list_new
            image_files_full += [file.replace('_M.png', '_I.png') for file in list_new]

        def get_category(mask_path):
            return mask_path.split('\\')[3].split('.')[0]

        def get_name(path):
            return path.split('\\')[3].split("avi")[0] + "avi"

        # RATE = 1
        self.df_full = pd.DataFrame({"image_path": [x for i, x in enumerate(image_files_full) if i % RATE == 0],
                                     "mask_path": [x for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                     "file_name": [get_name(x) for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                     "category": [get_category(x) for i, x in enumerate(mask_files_full) if
                                                  i % RATE == 0],
                                     "norm_name": ['#' + x.split('\\')[3].split(".")[0] for i, x in
                                                   enumerate(mask_files_full) if i % RATE == 0]})
        # print(self.df_full)

        return self.df_full

    def GetDatasetDataframeNew(self, RATE=1):
        mask_files_full = glob.glob(self.dirName + '*_M.png')
        image_files_full = [file.replace('_M.png', '_I.png') for file in mask_files_full]

        def get_category(mask_path):
            return mask_path.split('\\')[3].split('.')[0]

        def get_name(path):
            return path.split('\\')[3].split("avi")[0]+"avi"

        #RATE = 1
        self.df_full = pd.DataFrame({"image_path": [x for i, x in enumerate(image_files_full) if i % RATE == 0] ,
                                "mask_path": [x for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "file_name": [get_name(x) for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "category": [get_category(x) for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "norm_name" : ['#'+x.split('\\')[3].split(".")[0] for i, x in enumerate(mask_files_full) if i % RATE == 0]})
        #print(self.df_full)

        return self.df_full

    def GetDatasetDataframe(self, RATE=1):
        mask_files_full = glob.glob(self.dirName + '/*/M_*')
        image_files_full = [file.replace('M_', 'I_') for file in mask_files_full]

        def get_category(mask_path):
            return mask_path.split('\\')[1]

        def get_name(path):
            return path.split('\\')[2].replace('M_', '')

        #RATE = 1
        self.df_full = pd.DataFrame({"image_path": [x for i, x in enumerate(image_files_full) if i % RATE == 0] ,
                                "mask_path": [x for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "file_name": [get_name(x) for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "category": [get_category(x) for i, x in enumerate(mask_files_full) if i % RATE == 0],
                                "norm_name" : ["#"+x.split('\\')[2].split('_')[1]+"_"+x.split('\\')[2].split('_')[2].split('.')[0] for i, x in enumerate(mask_files_full) if i % RATE == 0]})
        #print(self.df_full)
        return self.df_full

    def GetSubsets(self, k=1):
        kfold_set = [0, 5, 10, 14, 18, 22]


        self.train_df_full = self.df_full[(self.df_full["norm_name"] != "2_4") & (self.df_full["norm_name"] != "22_2") &
                                          (self.df_full["norm_name"] != "1_4") & (self.df_full["norm_name"] != "3_21") &
                                          (self.df_full["norm_name"] != "17_3") & (self.df_full["norm_name"] != "19_3")]
        self.test_df_full = self.df_full[(self.df_full["norm_name"] == "2_4") | (self.df_full["norm_name"] == "22_2") |
                                         (self.df_full["norm_name"] == "1_4")]
        self.val_df_full = self.df_full[(self.df_full["norm_name"] == "3_21") | (self.df_full["norm_name"] == "17_3") |
                                        (self.df_full["norm_name"] == "19_3")]
        #print("FULL Train: {}\tVal: {}\tTest: {}".format(self.train_df_full.shape[0], self.test_df_full.shape[0], self.val_df_full.shape[0]))

        self.train_df_full = self.df_full[self.df_full["norm_name"].str.contains('#1_|#2_|#3_|#4_|#5_') == False]
        self.val_df_full = self.df_full[self.df_full["norm_name"].str.contains('#1_|#2_|#3_|#4_|#5_') == True]
        self.test_df_full = self.df_full[self.df_full["norm_name"].str.contains('#1_|#2_|#3_|#4_|#5_') == True]

    def PrepareDataloader(self):
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.Transpose(p=0.3),
            #A.GridDistortion(p=0.3),
            #A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.3)
        ])
        val_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
        ])
        test_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
        ])

        train_dataset = PupilDataset(self.train_df_full, train_transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                       collate_fn=my_collate)

        val_dataset = PupilDataset(self.val_df_full, val_transform)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                     collate_fn=my_collate)

        test_dataset = PupilDataset(self.test_df_full, test_transform)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                      collate_fn=my_collate)






# for PupilNet

import glob
import time
import pandas as pd
from fixein_traditional_method import *
import matplotlib.pyplot as plt

#image_files = glob.glob('X:\\Pupil Dataset\\PupilNet\\*\\*\\*.png')
#cp_files = glob.glob('X:\\Pupil Dataset\\PupilNet\\*\\*.txt')
image_files = glob.glob('X:\\Pupil Dataset\\Else\\*\\*.png')
cp_files = glob.glob('X:\\Pupil Dataset\\Else\\*.txt')

def get_category(image_path):
  return image_path.split('\\')[3]

def get_imagecount(image_path):
  return image_path.split('\\')[4].split('.')[0]

df = pd.DataFrame({"image_path": image_files,
                  "category": [get_category(x) for x in image_files],
                   "image_count": [get_imagecount(x) for x in image_files]
                   })
#print(df)
#print(cp_files)



def intTryParse(value):
    try:
        a = int(value)
        return True
    except ValueError:
        return False


class Stare_TM2(object):

    def __init__(self, dataset, cp_set):
        self.dataset = dataset
        self.cp_set = cp_set
        self.cp_dict = dict()
        self.create_cp_dict()

    def create_cp_dict(self):
        for cp in self.cp_set:
            prefix_name = cp.split('\\')[3]
            file = open(cp, "r")
            for l in file.readlines():
                cpsep = l.replace('\n', '').split(' ')
                try:
                    self.cp_dict["{0}_{1}".format(prefix_name, cpsep[1])] = "{0},{1}".format(cpsep[2], cpsep[3])
                except:
                    pass

    def run(self):
        th_methods = [define_thresholding1, define_thresholding2, define_thresholding3, define_thresholding4,
                      define_thresholding5, define_thresholding6, define_thresholding7, define_thresholding8,
                      define_thresholding9, define_thresholding10]
        fn_methods = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end]
        # dirName = "Results_20230128_2_{0}_{1}".format(finding_method.__name__, thresholding_method.__name__)
        # os.mkdir(dirName)
        for category in self.dataset["category"].unique():
            if category == "data set XIX": continue
            if (True):
                print("\n\nPARTICIPANT: {0}".format(category))
                for pd in fn_methods:
                    print("\t{0}".format(pd.__name__))
                    for th in th_methods:

                        list_acc = []
                        list_noteacc = []
                        list_size = []
                        list_exec = []
                        list_err = []
                        list_permacc = []

                        count = 0

                        for i, dsrow in self.dataset[(self.dataset["category"] == category)].iterrows():

                            image1 = cv2.imread(dsrow["image_path"])
                            mask1 = cv2.imread(dsrow["image_path"])
                            mask1[:, :, :] = 0
                            if not intTryParse(dsrow["image_count"]): continue
                            if category + "_corrected.txt_" + str(int(float(dsrow["image_count"]))) in self.cp_dict:
                                coordinate = self.cp_dict[category + "_corrected.txt_" + str(int(float(dsrow["image_count"])))]
                                coordinate = coordinate.split(",")
                                x, y = int(int(coordinate[0])/2), image1.shape[0]-int(int(coordinate[1])/2)
                                xcrop = int(image1.shape[1] * 0.05)
                                ycrop = int(image1.shape[0] * 0.05)
                                x -= xcrop
                                y -= ycrop
                                yshape = image1.shape[0]
                                xshape = image1.shape[1]
                                image = image1[ycrop:(yshape-ycrop), 0+xcrop:xshape-xcrop, :]
                                mask = mask1[ycrop:(yshape - ycrop), 0 + xcrop:xshape - xcrop, :]
                                xpercent = int(image.shape[1] * 0.04)
                                ypercent = int(image.shape[0] * 0.05)

                                mask_start_x, mask_end_x = x - xpercent, x + xpercent
                                mask_start_y, mask_end_y = y - ypercent, y + ypercent
                                if mask_start_x < 0: mask_start_x = 0
                                if mask_start_y < 0: mask_start_y = 0
                                if mask_end_x >= image.shape[1]: mask_end_x = image.shape[1]-1
                                if mask_end_y >= image.shape[0]: mask_end_y = image.shape[0]-1
                                mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x, :] = 255
                                #image[mask_start_y:mask_end_y, mask_start_x:mask_end_x, :] += 100
                                if False:
                                    fig = plt.figure()
                                    axarr = fig.add_subplot(1, 2, 1)
                                    axarr.imshow(mask)
                                    axarr3 = fig.add_subplot(1, 2, 2)
                                    axarr3.imshow(image)
                                    plt.show()
                                    continue

                                start_time = time.time()
                                image = image[:, :, 0]
                                mask = mask[:, :, 0]
                                minmax = procedure_min_max_simple(image)
                                smim = filter_savitzky_golay(minmax)
                                try:
                                    start_x, end_x, start_y, end_y, thx, thy = find_pupil_area(smim[0:2], pd, th)
                                except:
                                    continue
                                # select_pupil_region(smim[0:2])
                                end_time = time.time()

                                start_point = (start_x, start_y)
                                end_point = (end_x, end_y)
                                result = draw_rect_on_image(start_point, end_point,
                                                            smim, image, thx, thy)
                                resmask = draw_rect_on_image(start_point, end_point,
                                                             smim, mask, thx, thy)
                                acc, permission = calculate_mask_acc(start_point, end_point, mask)
                                note_acc = acc

                                if acc > 0.9 and False and count%100 == 0:
                                    fig = plt.figure()
                                    axarr = fig.add_subplot(1, 2, 1)
                                    axarr.imshow(result)
                                    axarr3 = fig.add_subplot(1, 2, 2)
                                    axarr3.imshow(image1)
                                    plt.show()

                                acc = 1 if acc > 0.4 else 0
                                err = 1 if acc < 0.1 else 0
                                size = (start_y - end_y) * (start_x - end_x) / (image1.size)

                                list_size.append(size)
                                list_acc.append(acc)
                                list_err.append(err)
                                list_noteacc.append(note_acc)
                                list_exec.append(end_time - start_time)
                                list_permacc.append(note_acc * permission)
                                #data["acc"] = acc
                                #data["err"] = err
                                #data["size"] = size
                                #data["extm"] = end_time - start_time

                                count += 1
                        print(
                            "\t\tAccr: {0:.4f}\tNacc: {2:.4f}\tPacc: {9:.4f}\tSize: {1:.4f}\tDacc: {5:.4f}\tGacc: {6:.4f}\tGnac: {7:.4f}\tAspc: {8:.4f}\tExec: {3:.4f}\tFreq: {4:.4f}".format(
                                np.mean(list_acc), np.mean(list_size), np.mean(list_noteacc), np.mean(list_exec),
                                1 / np.mean(list_exec), np.mean(list_acc) / np.mean(list_noteacc),
                                np.mean(list_acc) * np.mean(list_size), np.mean(list_noteacc) * np.mean(list_size),
                                np.abs(1 - np.mean(list_acc) / np.mean(list_noteacc)) * np.mean(list_acc) * np.mean(
                                    list_size), np.mean(list_permacc)), end="\n")



Stare_TM2(df, cp_files).run()
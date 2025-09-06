import time

import numpy as np
from matplotlib import pyplot as plt
from setuptools import glob
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
import sklearn


def get_all_sets(path):
    dfs = glob.glob(path + "\\*.xlsx")
    df = pd.DataFrame()
    for d in dfs:
        df = pd.concat([df, pd.read_excel(d)])
    return df


def plot_entropy_intensity_wrt_pixel_error(df):
    plot1 = plt.subplot2grid((1, 2), (0, 0))
    plot2 = plt.subplot2grid((1, 2), (0, 1))

    plot1.plot(df["pix_err_gt_pr"], df["entropy"], 'ro')
    plot1.set_title("Entropy")

    plot2.plot(df["pix_err_gt_pr"], df["intensity"], "ro")
    plot2.set_title("Intensity")

    plt.show()


def confusion_entropy_intensity_wrt_pixel_error(df):
    con_max = np.zeros((60, 60))
    for index, row in df.iterrows():
        entropy = 59 if row["entropy"] > 60 else int(row["entropy"])
        intensity = 59 if 255 - row["intensity"] > 59 else int(255 - row["intensity"])
        con_max[entropy][intensity] += 1 if int(row["pix_err_gt_pr_norm"]) > 5 else 0
    plt.imshow(con_max)
    plt.show()


def density(df):
    con_max = np.zeros((255, 255))
    for index, row in df.iterrows():
        con_max[int(row["entropy"])][int(row["intensity"])] += 1 if int(row["pix_err_gt_pr_norm"]) > 5 else 0

    plt.imshow(con_max + 10, extent=(np.amin(df["entropy"].astype("int")), np.amax(df["entropy"].astype("int")),
                                                      np.amin(df["intensity"].astype("int")), np.amax(df["intensity"].astype("int"))),
               cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.show()


def plot_3d_map(df):
    con_max_error = np.zeros((255, 255))
    for index, row in df.iterrows():
        con_max_error[int(row["entropy"])][int(row["intensity"])] += 1 if int(row["pix_err_gt_pr_norm"]) > 5 else 0

    con_max_success = np.zeros((255, 255))
    for index, row in df.iterrows():
        con_max_success[int(row["entropy"])][int(row["intensity"])] += 1 if int(row["pix_err_gt_pr_norm"]) <= 5 else 0


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(np.arange(0, 255), np.arange(0, 255), con_max_error, cmap='viridis')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Pixel Error')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(np.arange(0, 255), np.arange(0, 255), con_max_success, 50, cmap='viridis')
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Entropy')
    ax.set_zlabel('Pixel Error')
    plt.show()


def plot_changes_rate_x_y(df):
    categories = df["category"].unique()
    for c in categories:
        plt.plot(df[df["category"] == c]["pix_err_gt_pr_norm"] + 270, color='purple', label="perr")

        plt.plot(df[df["category"] == c]["entropy"], 'o', color="black", label="entropy")
        plt.plot(df[df["category"] == c]["intensity"], 'o', color="gray", label="intensity")

        plt.plot(df[df["category"] == c]["pr_cen_x"], "r", label="x_pr")
        plt.plot(df[df["category"] == c]["gt_cen_x"], "b", label="x_gt")

        plt.plot(df[df["category"] == c]["pr_cen_y"], "g", label="y_pr")
        plt.plot(df[df["category"] == c]["gt_cen_y"], "y", label="y_gt")

        plt.title(c)
        plt.legend()
        plt.show()


def plot_changes_rate_h_w(df):
    categories = df["category"].unique()
    for c in categories:
        plt.plot(df[df["category"] == c]["pix_err_gt_pr_norm"] + 270, color='purple', label="perr")

        plt.plot(df[df["category"] == c]["entropy"], 'o', color="black", label="entropy")
        plt.plot(df[df["category"] == c]["intensity"], 'o', color="gray", label="intensity")

        plt.plot(df[df["category"] == c]["pr_len_h"] * 4, "r", label="h_pr")
        plt.plot(df[df["category"] == c]["gt_len_h"] * 4, "b", label="h_gt")

        plt.plot(df[df["category"] == c]["pr_len_w"] * 4, "g", label="w_pr")
        plt.plot(df[df["category"] == c]["gt_len_w"] * 4, "y", label="w_gt")

        plt.title(c)
        plt.legend()
        plt.show()


def plot_changes_rate_ang(df):
    categories = df["category"].unique()
    for c in categories:
        plt.plot(df[df["category"] == c]["pix_err_gt_pr_norm"] + 270, color='purple', label="perr")

        plt.plot(df[df["category"] == c]["entropy"], 'o', color="black", label="entropy")
        plt.plot(df[df["category"] == c]["intensity"], 'o', color="gray", label="intensity")

        plt.plot(df[df["category"] == c]["pr_ang"] , "r", label="ang_pr")
        plt.plot(df[df["category"] == c]["gt_ang"] , "b", label="ang_gt")

        plt.title(c)
        plt.legend()
        plt.show()


def prepare_classes_for_entropy_and_intensity(df):
    # 0 success, 1 error
    mask = df["pix_err_gt_pr_norm"] <= 5
    df["class_entropy"] = 1
    df["class_intensity"] = 1
    df.loc[mask, "class_entropy"] = 0
    df.loc[mask, "class_intensity"] = 0
    return df


def plot_classes_for_entropy_and_intensity(df):
    categories = df["category"].unique()

    for c in categories:
        data_entropy = np.asarray(df[df["category"] == c]["entropy"])
        data_intensity = np.asarray(df[df["category"] == c]["intensity"])
        data_pix_err = np.asarray(df[df["category"] == c]["pix_err_gt_pr_norm"])
        list_error_entropy = list()
        list_error_intensity = list()
        list_error_index = list()

        plt.plot(data_pix_err + 270, color='purple', label="perr")

        plt.plot(data_entropy, 'o', color="black", label="entropy")
        plt.plot(data_intensity, 'o', color="gray", label="intensity")

        for i in range(0, len(data_pix_err) - 1):
            if data_pix_err[i] > 5:
                plt.axvline(x=i, color='yellow')
                list_error_index.append(i)
                list_error_entropy.append(data_entropy[i])
                list_error_intensity.append(data_intensity[i])

        plt.plot(list_error_index, list_error_entropy, "x", color="red", label="error_entropy")
        plt.plot(list_error_index, list_error_intensity, "x", color="blue", label="error_intensity")

        plt.title(c)
        plt.legend()
        plt.show()















def cluster_entropy_intensity_std(df):
    categories = df["category"].unique()
    for c in categories:
        data_entropy = np.asarray(df[df["category"] == c]["entropy"])
        data_intensity = np.asarray(df[df["category"] == c]["intensity"])
        data_pix_err = np.asarray(df[df["category"] == c]["pix_err_gt_pr_norm"])
        data_gt_x = np.asarray(df[df["category"] == c]["gt_cen_x"])
        data_gt_y = np.asarray(df[df["category"] == c]["gt_cen_y"])
        data_pr_x = np.asarray(df[df["category"] == c]["pr_cen_x"])
        data_pr_y = np.asarray(df[df["category"] == c]["pr_cen_y"])
        list_error_entropy = list()
        list_error_intensity = list()
        list_error_index = list()
        list_error_index2 = list()

        success, fail, miss, weak_success, block_fail = 0, 0, 0, 0, 0

        temp_df = df[df["category"] == c]
        list_entropy = df[df["category"] == c]["entropy"].tolist()
        list_intensity = df[df["category"] == c]["intensity"].tolist()
        list_pr_x = df[df["category"] == c]["pr_cen_x"].tolist()
        list_pr_y = df[df["category"] == c]["pr_cen_y"].tolist()

        interval = 121
        interval_coordinate = 12
        for i in range(interval, len(temp_df) - 1, 1):

            std_entropy = np.std(list_entropy[i - interval:i - 2])
            mean_entropy = np.mean(list_entropy[i - interval:i - 2])
            max_entropy = np.max(list_entropy[i - interval:i - 2])

            std_intensity = np.std(list_intensity[i - interval:i - 2])
            mean_intensity = np.mean(list_intensity[i - interval:i - 2])
            min_intensity = np.min(list_intensity[i - interval:i - 2])

            std_pr_x = np.std(list_pr_x[i - interval_coordinate:i - 1])
            mean_pr_x = np.mean(list_pr_x[i - interval_coordinate:i - 1])
            max_pr_x = np.min(list_pr_x[i - interval_coordinate:i - 1])

            std_pr_y = np.std(list_pr_y[i - interval_coordinate:i - 1])
            mean_pr_y = np.mean(list_pr_y[i - interval_coordinate:i - 1])
            max_pr_y = np.min(list_pr_y[i - interval_coordinate:i - 1])

            if max_entropy + std_entropy * 2 < list_entropy[i] or min_intensity - std_intensity * 2 > list_intensity[i]:
                '''if abs(mean_pr_x - list_pr_x[i]) > std_pr_x * 2 or abs(mean_pr_y - list_pr_y[i]) > std_pr_y * 2:
                    plt.axvline(x=i, color='black')
                    list_pr_x[i] = mean_pr_x
                    list_pr_y[i] = mean_pr_y
                else:
                    plt.axvline(x=i, color='pink')
                    fail += 1
                    list_error_index.append(i)
                    list_entropy[i] = mean_entropy
                    list_intensity[i] = mean_intensity'''

                plt.axvline(x=i, color='pink')
                fail += 1
                list_error_index.append(i)
                list_entropy[i] = mean_entropy
                list_intensity[i] = mean_intensity
                # 700 deg/sec -> 5.833 deg per frame
                # 50 deg ROI from 4 cms: 3.69399
                # 5.833  deg ROI from 4: 0.40704
                # percent              : 0.11018
                # x is 320 px          : 35.2576 -> 35
                # y is 240 px          : 26.4432 -> 26
                if abs(list_pr_x[i-1] - list_pr_x[i]) <= 35 and  \
                   abs(list_pr_y[i-1] - list_pr_y[i]) <= 26:
                    plt.axvline(x=i, color='black')
                    list_pr_x[i] = mean_pr_x
                    list_pr_y[i] = mean_pr_y
                    block_fail += 1




        plt.plot(data_entropy, 'o', color="black", label="entropy")
        plt.plot(data_intensity, 'o', color="gray", label="intensity")



        for i in range(0, len(data_pix_err) - 1):
            if data_pix_err[i] >= 4.5:
                if i in list_error_index:
                    plt.axvline(x=i, color='green', linestyle="dashed")
                    success, fail = success + 1, fail - 1
                else:
                    plt.axvline(x=i, color='yellow', linestyle="dashed")
                    miss += 1
                list_error_index2.append(i)
                list_error_entropy.append(data_entropy[i])
                list_error_intensity.append(data_intensity[i])
            elif data_pix_err[i] >= 1.5:
                if i in list_error_index:
                    plt.axvline(x=i, color='turquoise', linestyle="dashed")
                    weak_success, fail = weak_success + 1, fail - 1

        plt.plot(list_error_index2, list_error_entropy, "x", color="red", label="error_entropy")
        plt.plot(list_error_index2, list_error_intensity, "x", color="blue", label="error_intensity")
        plt.plot(data_pix_err + 270, color='purple', label="perr")

        plt.plot(data_pr_x, "orange", label="x_pr")
        plt.plot(data_gt_x, "mediumvioletred", label="x_gt")

        plt.plot(data_pr_y, "g", label="y_pr")
        plt.plot(data_gt_y, "y", label="y_gt")


        print(c + " (s:" + str(success) + ", f:" + str(fail) + ", m:" + str(miss) + ", ws:" + str(weak_success) + ")")

        plt.title(c + " (s:" + str(success) + ", f:" + str(fail) + ", m:" + str(miss) + ", ws:" +
                  str(weak_success) +  ", bf:" + str(block_fail) +  ")\n " +
                  "Base Acc: " + str(1 - (success + miss) / len(list_entropy)) +
                  "      Max Acc: " + str(1 - (miss / len(list_entropy))))
        plt.legend()
        plt.show()



#df = get_all_sets("X:\\Fixein Pupil Detection\\SegmentedEllipsePrediction\\1680883969")
#df.to_csv("X:\\Fixein Pupil Detection\\SegmentedEllipsePrediction\\1680883969\\all_1680883969.csv")

df = pd.read_csv("X:\\Fixein Pupil Detection\\SegmentedEllipsePrediction\\1680883969\\all_1680883969.csv")
df = prepare_classes_for_entropy_and_intensity(df)

#plot_entropy_intensity_wrt_pixel_error(df)
#confusion_entropy_intensity_wrt_pixel_error(df)
#density(df)
#plot_3d_map(df)
#plot_changes_rate_x_y(df)
#plot_changes_rate_h_w(df)
#plot_changes_rate_ang(df)
#plot_classes_for_entropy_and_intensity(df)
cluster_entropy_intensity_std(df)


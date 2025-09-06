import math

import numpy as np
import scipy
from matplotlib import pyplot as plt
from setuptools import glob
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import seaborn as sn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import IsolationForest
import sklearn
import time


def prepare_classes_for_entropy_and_intensity(df):
    # 0 success, 1 error
    mask = df["pix_err_gt_pr_norm"] <= 5
    df["class_entropy"] = 1
    df["class_intensity"] = 1
    df.loc[mask, "class_entropy"] = 0
    df.loc[mask, "class_intensity"] = 0
    return df



class EllipseTrendApproximation(object):
    def __init__(self, df):
        self.df = df
        self.interval_ei = 61
        self.interval_ei_correct = 61
        self.interval_el = 10
        self.approximate_trend_base(df)
        self.sup_to_cor = True
        self.last_success_x, self.last_success_y = 0, 0

    def filter_savitzky_golay(self, data):
        def odd_condition(val): return val + 1 if val % 2 == 0 else val
        winl = odd_condition(len(data)-1)
        return scipy.signal.savgol_filter(data, winl, 1)

    def approximate_trend(self, i):
        self.interval_ei = self.interval_ei_correct if i > self.interval_ei_correct else i
        temp_e = self.data_entropy_changed[i - self.interval_ei:i]
        temp_i = self.data_intensity_changed[i - self.interval_ei:i]

        if (abs(self.data_pr_x[i] - self.data_pr_x_changed[i - 1]) >= 35 or
                abs(self.data_pr_y[i] - self.data_pr_y_changed[i - 1]) >= 26) and \
            (np.mean(temp_e) + max(np.std(temp_e) * 2, 1) <= self.data_entropy[i] or
             np.mean(temp_i) - max(np.std(temp_i) * 2, 1) >= self.data_intensity[i]):
            self.max_change_error.append(i)
            '''
            self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                np.mean(self.data_pr_x_changed[i - 7: i - 2]), np.mean(self.data_pr_y_changed[i - 7: i - 2])
            self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                np.mean(self.data_entropy_changed[i - 7: i - 2]), np.mean(self.data_intensity_changed[i - 7: i - 2])
            '''
            self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                self.data_pr_x_changed[i - 1], self.data_pr_y_changed[i - 1]
            self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                self.data_entropy_changed[i - 1], self.data_intensity_changed[i - 1]


            #self.last_success_x, self.last_success_y = \
            #    self.data_pr_x_changed[i], self.data_pr_y_changed[i]
            #self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
            #    self.last_success_x, self.last_success_y

            self.sup_to_cor = False

        elif i >= self.interval_ei: #and self.data_intensity[i] != 0:

            #if np.mean(temp_e) + np.std(temp_e) * 2 <= self.data_entropy[i]: self.entropy_error.append(i)
            #if np.mean(temp_i) - np.std(temp_i) * 2 >= self.data_intensity[i]: self.intensity_error.append(i)

            if self.data_intensity[i] < 128 and False:
                self.intensity_error.append(i)
                self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                    self.last_success_x, self.last_success_y
                self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                    self.data_entropy_changed[i - 1], self.data_intensity_changed[i - 1]

            elif (np.mean(temp_e) + max(np.std(temp_e) ** 2, 1) <= self.data_entropy[i] and
                np.mean(temp_i) - max(np.std(temp_i) ** 2, 1) >= self.data_intensity[i]):
                self.intensity_error.append(i)

            elif (np.mean(temp_e) + max(np.std(temp_e) * 2, 1) <= self.data_entropy[i] and
                np.mean(temp_i) - max(np.std(temp_i) * 2, 1) >= self.data_intensity[i])\
                    or self.data_intensity[i] < 128:# or \
                #np.mean(temp_e) + max(np.std(temp_e) * 2, 26) <= self.data_entropy[i] or \
                #np.mean(temp_i) - max(np.std(temp_i) * 2, 26) >= self.data_intensity[i]:
                self.ei_error.append(i)
                '''
                self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                    self.data_pr_x_changed[i - 1], self.data_pr_y_changed[i - 1]
                self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                    self.data_entropy_changed[i - 1], self.data_intensity_changed[i - 1]
                
                
                #self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                #    self.filter_savitzky_golay(self.data_entropy_changed[i-self.interval_ei:i])[-1], \
                #    self.filter_savitzky_golay(self.data_intensity_changed[i - self.interval_ei:i])[-1]
                

                self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                    self.filter_savitzky_golay(self.data_pr_x_changed[i - self.interval_ei:i])[-1], \
                    self.filter_savitzky_golay(self.data_pr_y_changed[i - self.interval_ei:i])[-1]
                '''
                self.data_entropy_changed[i], self.data_intensity_changed[i] = \
                    self.data_entropy_changed[i - 1], self.data_intensity_changed[i - 1]

                if False and self.sup_to_cor and \
                        (np.mean(temp_e) + max(np.std(temp_e) * 3, 1) <= self.data_entropy[i] and
                         np.mean(temp_i) - max(np.std(temp_i) * 3, 1) >= self.data_intensity[i]):
                    self.data_pr_x_changed[i], self.data_pr_y_changed[i] = \
                        self.filter_savitzky_golay(self.data_pr_x_changed[i - 10:i])[-1], \
                        self.filter_savitzky_golay(self.data_pr_y_changed[i - 10:i])[-1]
                    self.sup_to_cor = False
                else: self.sup_to_cor = True

            else:
                self.last_success_x, self.last_success_y = \
                    self.data_pr_x_changed[i], self.data_pr_y_changed[i]
                self.sup_to_cor = False





    def approximate_trend_base(self, df):
        categories = df["category"].unique()
        for c in categories:
            self.c = c
            self.data_entropy = np.asarray(df[df["category"] == c]["entropy"])
            self.data_intensity = np.asarray(df[df["category"] == c]["intensity"])
            self.data_entropy_changed = np.asarray(df[df["category"] == c]["entropy"])
            self.data_intensity_changed = np.asarray(df[df["category"] == c]["intensity"])
            self.data_pix_err = np.asarray(df[df["category"] == c]["pix_err_gt_pr_norm"])
            self.data_gt_x = np.asarray(df[df["category"] == c]["gt_cen_x"])
            self.data_gt_y = np.asarray(df[df["category"] == c]["gt_cen_y"])
            self.data_pr_x = np.asarray(df[df["category"] == c]["pr_cen_x"])
            self.data_pr_y = np.asarray(df[df["category"] == c]["pr_cen_y"])
            self.data_pr_x_changed = np.asarray(df[df["category"] == c]["pr_cen_x"])
            self.data_pr_y_changed = np.asarray(df[df["category"] == c]["pr_cen_y"])

            self.success, self.fail, self.miss, self.weak_success, self.block_fail = 0, 0, 0, 0, 0

            self.entropy_error = []
            self.intensity_error = []
            self.ei_error = []
            self.max_change_error = []
            for i in range(1, len(self.data_entropy)): self.approximate_trend(i)

            self.generate_new_pixel_error_rate()
            self.plot_general_values()



    def plot_general_values(self):
        total = len(self.data_pix_err)
        normal_count, trend_count = (self.data_pix_err <= 5).sum(), (np.asarray(self.new_pix_err) <= 5).sum()
        normal = (self.data_pix_err <= 5).sum() / total
        trend = (np.asarray(self.new_pix_err) <= 5).sum() / total

        if trend < 0.8 or trend - normal > 0.02 or trend < normal:
            if trend < normal:
                print("\n[X]" + self.c)
            else:
                print("\n[+]" + self.c)
            print("NORMAL: " + str(normal))
            print("TREND : " + str(trend))
            print("GAIN: ", str(trend_count - normal_count), ", NT:", str(normal_count), " - TT:", str(trend_count), ", NF:",
                  str(total - normal_count), " - TF:", str(total - trend_count))

            np_tp, np_tn, nn_tp, nn_tn = 0, 0, 0, 0
            for i in range(0, len(self.data_pix_err)-1):
                if self.data_pix_err[i] <= 5 and self.new_pix_err[i] <= 5: np_tp += 1
                if self.data_pix_err[i] <= 5 and self.new_pix_err[i] > 5: np_tn += 1
                if self.data_pix_err[i] > 5 and self.new_pix_err[i] <= 5: nn_tp += 1
                if self.data_pix_err[i] > 5 and self.new_pix_err[i] > 5: nn_tn += 1
            print("NP_TP: ", np_tp, ", NP_TN: ", np_tn, ", NN_TP: ", nn_tp, ", NN_TN:", nn_tn)
            print("NORM GAIN: ", str(1-(total-trend_count)/(total-normal_count)))

            plt.plot(self.data_entropy_changed, 'o', color="blue", label="ent_cor")
            plt.plot(self.data_intensity_changed, 'o', color="red", label="int_cor")

            plt.plot(self.data_entropy, 'o', color="black", label="entropy")
            plt.plot(self.data_intensity, 'o', color="gray", label="intensity")





            for i in self.intensity_error: plt.axvline(x=i, color='blue', linestyle="solid")
            for i in self.entropy_error: plt.axvline(x=i, color='red', linestyle="dashed")
            for i in self.ei_error: plt.axvline(x=i, color='green', linestyle="solid")
            for i in self.max_change_error: plt.axvline(x=i, color='black', linestyle="dotted")

            plt.plot(self.data_pr_y, "g", label="y_pr")
            plt.plot(self.data_gt_y, "y", label="y_gt")
            plt.plot(self.data_pr_y_changed, "darkblue", label="y_pr_changed")

            plt.plot(self.data_pr_x, "orange", label="x_pr")
            plt.plot(self.data_gt_x, "mediumvioletred", label="x_gt")
            plt.plot(self.data_pr_x_changed, "blue", label="x_pr_changed")

            def draw_sg_lines(data, size, name):
                output = list()
                for i in range(0, len(data)-1):
                    if i>10:
                        a = size if i >= size else i
                        sg_normal_x = self.filter_savitzky_golay(data[i-a: i+1])
                        output.append(sg_normal_x[-1])
                    else: output.append(0)
                plt.plot(np.asarray(output), "red", label=name, linestyle="dashed")
            #draw_sg_lines(self.data_pr_x_changed, 5, "x_pr_sg")
            #draw_sg_lines(self.data_pr_y_changed, 5, "y_pr_sg")

            accuracy = list()
            for i in range(0, len(self.data_entropy)):
                acc = (1 - (self.data_entropy[i] / 255) + (self.data_intensity[i] / 255)) / 2
                accuracy.append(acc * 20)

            plt.axhline(y=290, color='gray', linestyle="solid")
            plt.axhline(y=320, color='gray', linestyle="solid")
            plt.axhline(y=350, color='gray', linestyle="solid")
            plt.axhspan(270, 290, facecolor='black', alpha=0.2)
            plt.axhspan(300, 320, facecolor='black', alpha=0.2)
            plt.axhspan(330, 350, facecolor='black', alpha=0.2)
            plt.axhline(y=275, color='yellow', linestyle="solid")
            plt.axhline(y=305, color='yellow', linestyle="solid")
            plt.plot(self.data_pix_err + 270, color='purple', label="perr")
            plt.plot(np.asarray(self.new_pix_err) + 300, color='magenta', label="perr")
            plt.plot(np.asarray(accuracy) + 330, color='cyan', label="acc")
            plt.axhline(y=270, color='black', linestyle="solid")
            plt.axhline(y=300, color='black', linestyle="solid")
            plt.axhline(y=330, color='black', linestyle="solid")

            plt.legend()
            plt.show()

        #plt.show()

    def generate_new_pixel_error_rate(self):
        self.new_pix_err = list()
        for i in range(len(self.data_pr_y_changed)):
            new_error = math.dist(
                (self.data_gt_x[i], self.data_gt_y[i]),
                (self.data_pr_x_changed[i], self.data_pr_y_changed[i]) )
            self.new_pix_err.append(new_error if new_error < 20 else 20)


df = pd.read_csv("X:\\Fixein Pupil Detection\\SegmentedEllipsePrediction\\1680883969\\all_1680883969.csv")
df = prepare_classes_for_entropy_and_intensity(df)

analysis = EllipseTrendApproximation(df)
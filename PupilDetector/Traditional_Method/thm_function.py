import numpy as np

class THM_Functions(object):

  @staticmethod
  def define_thresholding1(d):
    return min(np.average(d), (np.max(d) + np.min(d)) * 0.5)

  @staticmethod
  def define_thresholding2(d):
    return max(np.average(d), (np.max(d) + np.min(d)) * 0.5)

  @staticmethod
  def define_thresholding4(d):
    return np.average(d)

  @staticmethod
  def define_thresholding6(d):
    m = np.min(d)
    return ((np.max(d) - m) * 0.33) + m

  @staticmethod
  def define_thresholding7(d):
    m = np.min(d)
    return ((np.max(d) - m) * 0.66) + m

  @staticmethod
  def define_thresholding8(d):
    return (np.average(d) + np.min(d)) * 0.5

  @staticmethod
  def define_thresholding9(d):
    return np.max(d) - np.std(d)

  @staticmethod
  def define_thresholding10(d):
    return np.min(d) + np.std(d)

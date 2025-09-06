import cv2
import numpy as np
import scipy
from scipy.stats import skew, kurtosis


def define_thresholding1(d):
  return min(np.average(d), (np.max(d) + np.min(d))  * 0.5)

def define_thresholding2(d):
  return max(np.average(d), (np.max(d) + np.min(d))  * 0.5)

def define_thresholding3(d):
  return (np.max(d) + np.min(d)) * 0.5

def define_thresholding4(d):
  return np.average(d)

def define_thresholding5(d):
  return (np.max(d) + np.min(d)) * 0.66

def define_thresholding6(d):
  m = np.min(d)
  return ((np.max(d) - m) * 0.33) + m

def define_thresholding7(d):
  m = np.min(d)
  return ((np.max(d) - m) * 0.66) + m

def define_thresholding8(d):
  return (np.average(d) + np.min(d)) * 0.5

def define_thresholding9(d):
  return np.max(d) - np.std(d)

def define_thresholding10(d):
  return np.min(d) + np.std(d)









def find_pupil_area_med_first_old(d, th):
  left, right, leftp, rightp, mlen = False, False, 0, 0, int(len(d)/2)
  for i in range(0,mlen):
    if d[mlen-i] < th or d[mlen+i] < th:
      leftp, rightp, meanp = i, i, mlen-i if d[mlen-i] < th else mlen+i
      for j in range(0, mlen-i):
        if not left:
          if d[meanp-j] < th: leftp = meanp-j
          else: left = True
        if not right:
          if d[meanp+j] < th: rightp = meanp+j
          else: right = True
        if left and right: return leftp, rightp
      return leftp, rightp


def find_pupil_area_med_first(d, th):
  mid_index = int(len(d)/2)
  min_indexes = np.where(d < th)[0] - mid_index
  center_min_index = min_indexes[np.abs(min_indexes).argmin()] + mid_index
  border_end_indexes = np.where(d[center_min_index: len(d)-1] >= th)[0]
  border_start_indexes = np.where(d[0: center_min_index+1] >= th)[0]
  leftp, rightp = 0, len(d)-1
  if len(border_start_indexes) != 0:
    leftp = border_start_indexes[-1]
  if len(border_end_indexes) != 0:
    rightp = border_end_indexes[0] + center_min_index
  return leftp, rightp





def find_pupil_area_max_length_old(d, th):
  selected, temp, lowerbound, mlen = [0,0], 0, False, len(d)
  for i in range(0,mlen):
    if d[i] < th and not lowerbound:
      temp = i
      lowerbound = True
    elif d[i] > th and lowerbound:
      if selected[1]-selected[0]<i-temp:
        selected = [temp, i]
        temp = mlen
      else: temp = i
      lowerbound = False
  if selected[1]-selected[0] > mlen-temp: return selected[0], selected[1]
  else: return temp, mlen

def find_pupil_area_max_length(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 0, [0, 0]
  for arr in splited_mins:
    if len(arr) > i:
      i, selected_arr = len(arr), arr
  return selected_arr[0], selected_arr[-1]



def find_pupil_area_max_depth(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 255, [0, 0]
  for arr in splited_mins:
    if len(arr) > 1:
      score = d[arr[0]: arr[-1]].min()
      if score < i:
        i, selected_arr = score, arr
  return selected_arr[0], selected_arr[-1]


def find_pupil_area_max_length_and_depth_product(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 0, [0, 0]
  for arr in splited_mins:
    if len(arr) > 1:
      score = len(arr) * (th - d[arr[0]: arr[-1]].min())
      if score > i:
        i, selected_arr = score, arr
  return selected_arr[0], selected_arr[-1]



def find_pupil_area_skewness(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 255, [0, 0]
  for arr in splited_mins:
    if len(arr) > 1:
      skewness = abs(skew(d[arr[0]: arr[-1]], axis=0, bias=True))
      if skewness < i:
        i, selected_arr = skewness, arr
  return selected_arr[0], selected_arr[-1]

def find_pupil_area_kurtosis(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 255, [0, 0]
  for arr in splited_mins:
    if len(arr) > 1:
      kurtosis_value = abs(kurtosis(d[arr[0]: arr[-1]], axis=0, bias=True) - 3)
      if kurtosis_value < i:
        i, selected_arr = kurtosis_value, arr
  return selected_arr[0], selected_arr[-1]

def find_pupil_area_skewness_and_kurtosis(d, th):
  min_indexes = np.where(d < th)[0]
  splited_mins = np.split(min_indexes, np.where(np.diff(min_indexes) != 1)[0] + 1)
  if len(splited_mins) == 1: return splited_mins[0][0], splited_mins[0][-1]
  i, selected_arr = 255, [0, 0]
  for arr in splited_mins:
    if len(arr) > 1:
      score = abs(skew(d[arr[0]: arr[-1]], axis=0, bias=True)) + abs(kurtosis(d[arr[0]: arr[-1]], axis=0, bias=True) - 3)
      if score < i:
        i, selected_arr = score, arr
  return selected_arr[0], selected_arr[-1]




def find_pupil_area_first_and_end_old(d, th):
  firstp, lastp, mlen = 0, 0, len(d)
  for i in range(0, mlen):
    if d[i]<th:
      if firstp == 0: firstp = i
      else: lastp = i
  return firstp, lastp

def find_pupil_area_first_and_end(d, th):
  min_indexes = np.where(d < th)[0]
  if len(min_indexes) >= 2: return min_indexes[0], min_indexes[-1]
  else: return 0, 0

def find_pupil_area(d, finding_method, thresholding_method):
  thx = thresholding_method(d[0])
  thy = thresholding_method(d[1])
  start_x, end_x = finding_method(d[0], thx)
  start_y, end_y = finding_method(d[1], thy)
  return start_x, end_x, start_y, end_y, thx, thy

def find_pupil_area_axes(d, finding_method_x, finding_method_y, thresholding_method):
  thx = thresholding_method(d[0])
  thy = thresholding_method(d[1])
  try:
    start_x, end_x = finding_method_x(d[0], thx)
    start_y, end_y = finding_method_y(d[1], thy)
  except:
    start_x, start_y, end_x, end_y = 0, 0, 0, 0
  return start_x, end_x, start_y, end_y, thx, thy


def filter_savitzky_golay(data):
  def odd_condition(val): return val + 1 if val % 2 == 0 else val

  winl_x, winl_y = odd_condition(int(len(data[0]) * 0.20)), odd_condition(int(len(data[2]) * 0.20))
  ply_x, ply_y = 2, 2
  return [scipy.signal.savgol_filter(data[0], winl_x, ply_x), scipy.signal.savgol_filter(data[1], winl_y, ply_y), \
          scipy.signal.savgol_filter(data[2], winl_x, ply_x), scipy.signal.savgol_filter(data[3], winl_y, ply_y)]

def filter_savitzky_golay_to_be_used(data, rate=0.2):
  def odd_condition(val): return val + 1 if val % 2 == 0 else val
  winl_x, winl_y = odd_condition(int(len(data[0]) * rate)), odd_condition(int(len(data[1]) * rate))
  ply_x, ply_y = 2, 2
  return [scipy.signal.savgol_filter(data[0], winl_x, ply_x), scipy.signal.savgol_filter(data[1], winl_y, ply_y)]



def procedure_min_max_simple(data, last_pixel=1):
  newdatax = np.copy(data)
  newminx = np.zeros((data.shape[0], last_pixel))
  newdatay = np.copy(data)
  newminy = np.zeros((data.shape[1], last_pixel))

  for i in range(0, last_pixel):
    newminx[:, i] = newdatax[np.arange(len(newdatax)), np.argmin(newdatax, axis=1)]
    newdatax[np.arange(len(newdatax)), np.argmin(newdatax, axis=1)] = 255

    newminy[:, i] = newdatay[np.argmin(newdatay, axis=0), np.arange(newminy.shape[0])]
    newdatay[np.argmin(newdatay, axis=0), np.arange(newminy.shape[0])] = 255

  return [np.average(newminy, axis=1), np.average(newminx, axis=1), \
          np.max(data, axis=0), np.max(data, axis=1)]

  return [np.min(data, axis=0), np.min(data, axis=1), \
          np.max(data, axis=0), np.max(data, axis=1)]


def procedure_min_max_simple_to_be_used(data):
  return [data[np.argmin(data, axis=0), np.arange(data.shape[1])],
          data[np.arange(len(data)), np.argmin(data, axis=1)]]

'''
  newminx = np.zeros((data.shape[0], 1))
  newminy = np.zeros((data.shape[1], 1))

  newminx[:, 0] = data[np.arange(len(data)), np.argmin(data, axis=1)]
  newminy[:, 0] = data[np.argmin(data, axis=0), np.arange(newminy.shape[0])]

  return newminy, newminx
  #return [np.min(newminy, axis=1), np.min(newminx, axis=1)]
'''



def select_pupil_region(data):
  def find_start_and_end(d, fromoutset=True):
    threshold = min(np.average(d), (np.max(d) + np.min(d)) / 2)
    # lower_threshold = ((np.max(d) + np.min(d)) /4) + np.min(d) +1
    array_main, ref_point = int(len(d) / 2) - 1, -1
    lower_bound, higher_bound, lower_control, higher_control = 0, 0, True, True

    rang = np.arange(int(len(d) * 0.0), int(len(d) * 1.0))
    temp = 0
    pupil_area_width = 0
    pupil_area_control = False
    if not fromoutset: rang = rang[::-1]
    for i in rang:
      temp = i
      if ref_point == -1 and d[i] < threshold:
        ref_point = i
        lower_bound, higher_bound = ref_point, ref_point
        pupil_area_width += 1
        pupil_area_control = True

      if ref_point > -1 and d[i] < threshold:
        pupil_area_width += 1
        if pupil_area_width > 45:
          if fromoutset:
            higher_bound = i
          else:
            lower_bound = i
        # return lower_bound, higher_bound, threshold
      elif d[i] > threshold:
        pupil_area_control = False
        pupil_area_width = 0

    if pupil_area_control:
      if fromoutset:
        higher_bound = temp
      else:
        lower_bound = temp
    return lower_bound, higher_bound, threshold

  start_x, end_x, thx = find_start_and_end(data[0])
  start_y, end_y, thy = find_start_and_end(data[1], False)

  return start_x, end_x, start_y, end_y, thx, thy


def draw_rect_on_image(start_point, end_point, pdata, im, thx, thy):
  im = np.stack((im,) * 3, axis=-1)
  color = 0
  thickness = 4
  im = np.ascontiguousarray(im, dtype=np.uint8)

  x = np.arange(len(pdata[0]))
  y = np.arange(len(pdata[1]))
  xpts = np.vstack((x, pdata[0])).astype(np.int32).T
  ypts = np.vstack((pdata[1], y)).astype(np.int32).T

  cv2.polylines(im, [xpts], isClosed=False, color=(255, 0, 0))
  cv2.polylines(im, [ypts], isClosed=False, color=(0, 255, 0))

  cv2.line(im, (0, 240), (640, 240), color=(255, 0, 0))
  cv2.line(im, (320, 0), (320, 640), color=(0, 255, 0))

  thx, thy = int(thx), int(thy)
  cv2.line(im, (0, thx), (640, thx), color=(255, 0, 0))
  cv2.line(im, (thy, 0), (thy, 640), color=(0, 255, 0))

  return cv2.rectangle(im, start_point, end_point, color, thickness)


def calculate_mask_acc(start_point, end_point, mask):
    mask = mask / 255.
    total_area = np.count_nonzero(mask > 0.03)
    found_area = np.count_nonzero(mask[start_point[1]:end_point[1], start_point[0]:end_point[0]] > 0.03)
    if total_area == 0:
      total_area, found_area = 1, 1
    # if (found_area/total_area < 0.3): print("ERROR: "+str(found_area)+" / "+str(total_area))
    if total_area == 0: return 1, 1

    main_y = int((start_point[1] + end_point[1]) / 2)
    main_x = int((start_point[0] + end_point[0]) / 2)
    control = 0
    permrate = 0.05
    totalcount = mask[start_point[1]:main_y, start_point[0]:main_x].size
    if (totalcount - np.count_nonzero(
      mask[start_point[1]:main_y, start_point[0]:main_x] > 0.03) > totalcount * permrate): control += 1
    if (totalcount - np.count_nonzero(
      mask[main_y:end_point[1], start_point[0]:main_x] > 0.03) > totalcount * permrate): control += 1
    if (totalcount - np.count_nonzero(
      mask[start_point[1]:main_y, main_x:end_point[0]] > 0.03) > totalcount * permrate): control += 1
    if (totalcount - np.count_nonzero(
      mask[main_y:end_point[1], main_x:end_point[0]] > 0.03) > totalcount * permrate): control += 1
    return found_area / total_area, 1 if control > 1 else 0

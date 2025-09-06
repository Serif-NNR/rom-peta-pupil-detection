import scipy, cv2
import numpy as np
import skimage.exposure as exposure

def filter_savitzky_golay(data, rate=0.2):
  def odd_condition(val): return val + 1 if val % 2 == 0 else val
  winl_x, winl_y = odd_condition(int(len(data[0]) * rate)), odd_condition(int(len(data[1]) * rate))
  ply_x, ply_y = 2, 2
  return [scipy.signal.savgol_filter(data[0], winl_x, ply_x), scipy.signal.savgol_filter(data[1], winl_y, ply_y)]

def get_sobel_edge(map):
    sobelx = cv2.Sobel(map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(map, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.multiply(sobelx, sobelx)
    sobely2 = cv2.multiply(sobely, sobely)
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    edge_im = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)).clip(0, 255).astype(
      np.uint8)
    return edge_im

def get_ellipse_parameters(map):
    thxxx, threshed = cv2.threshold(map, 25, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selectedcnt, ellipse = None, [[0, 0], [0, 0], 0]
    cntcount = 0
    for cnt in cnts:
        if cnt.shape[0] > 5 and cntcount < cnt.shape[0]:
            cntcount = cnt.shape[0]
            selectedcnt = cnt
    if cntcount > 0: ellipse = cv2.fitEllipse(selectedcnt)
    return ellipse


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



def draw_rect_on_image(start_point, end_point, pdata, im, thx, thy):
  im = np.stack((im,) * 3, axis=-1)
  color = (180, 180, 0)
  thickness = 4
  thickness_line = 2
  im = np.ascontiguousarray(im, dtype=np.uint8)

  x = np.arange(len(pdata[0]))
  y = np.arange(len(pdata[1]))
  xpts = np.vstack((x, pdata[0])).astype(np.int32).T
  ypts = np.vstack((pdata[1], y)).astype(np.int32).T

  cv2.polylines(im, [xpts], False, (255, 0, 0), 1)
  cv2.polylines(im, [ypts], False, (0, 255, 0), 1)

  #cv2.line(im, (0, 240), (640, 240), (155, 0, 0), thickness_line)
  #cv2.line(im, (320, 0), (320, 640), (0, 155, 0), thickness_line)

  thx, thy = int(thx), int(thy)
  cv2.line(im, (0, thx), (640, thx), (155, 0, 0), thickness_line)
  cv2.line(im, (thy, 0), (thy, 640), (0, 155, 0), thickness_line)

  return cv2.rectangle(im, start_point, end_point, color, thickness)

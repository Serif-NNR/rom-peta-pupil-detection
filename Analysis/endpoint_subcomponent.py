import time, cv2, torch
from skimage.exposure import exposure
from fixein_traditional_method import *


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


pf_method = [find_pupil_area_med_first, find_pupil_area_max_length, find_pupil_area_first_and_end,
             find_pupil_area_max_depth, find_pupil_area_max_length_and_depth_product]
th_method_base = define_thresholding8
th_method = [define_thresholding1, define_thresholding2,
             define_thresholding4, define_thresholding6,
             define_thresholding7, define_thresholding8, define_thresholding9,
             define_thresholding10]
def get_auto_operaton():
    selected_pfx, selected_pfy, score = None, None, 0
    for pfx in pf_method:
        for pfy in pf_method:
            start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, pfx, pfy, th_method_base)
            size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / resolution)
            acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), map)
            tscore = size_rate * min(acc)
            if tscore > score: selected_pfx, selected_pfy, score = pfx, pfy, tscore

    selected_thm, score = None, 0
    for thm in th_method:
        start_x, end_x, start_y, end_y, thx, thy = find_pupil_area_axes(smim, selected_pfx, selected_pfy, thm)
        size_rate = 1 - (((end_y - start_y) * (end_x - start_x)) / resolution)
        acc = calculate_mask_acc((start_x, start_y), (end_x, end_y), map)
        tscore = size_rate * min(acc)
        if tscore > score: selected_thm, score = thm, tscore
    return score

def get_entropy_and_intensity(feature_map, center, length, angle):
    center, length, angle = [int(center[0]), int(center[1])], [int(length[0]/2), int(length[1]/2)], int(angle)
    maxl = int(max(length))
    psx, pex, psy, pey = max(center[0] - maxl, 0), center[0] + maxl, max(center[1] - maxl, 0), center[1] + maxl
    map = feature_map[psy:pey, psx:pex]
    mask_center = (center[0] - psx, center[1] - psy)
    mask = np.zeros(map.shape)
    mask = cv2.ellipse(mask, mask_center, length, angle, 0, 360, 1, -1)
    mask = cv2.ellipse(mask, mask_center, length, angle, 0, 360, 0, 2)

    total_pixel_count = (mask > 0).sum()
    edge_map = get_sobel_edge(map)
    intensity_map, enropy_map = map * mask, edge_map * mask

    intensity = np.sum(intensity_map[mask > 0]) / total_pixel_count
    entropy = np.sum(enropy_map[mask > 0]) / total_pixel_count

    return entropy, intensity


def convert_matrx_to_tensor():
    temp = np.expand_dims(patch_tr, axis=-1).transpose((2, 0, 1))
    temp = torch.tensor(temp.astype(np.float32) / 255.).unsqueeze(0)
    return temp

def convert_tensor_to_matrx():
    return (temp2.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('{0}: {1}'.format(method.__name__, str((te-ts)/1000)))
        return result
    return timed


resolution = 320 * 240
mg = np.random.rand(320, 240)
patch_mg = np.random.rand(32, 24)
patch_tr = cv2.imread("X:\\Pupil Dataset\\Dikablis_Full\\dikablisR_1_1_0_M.png")[0:120, 213:320, 0]
map = cv2.imread("X:\\Pupil Dataset\\Dikablis_Full\\dikablisR_1_1_0_M.png")[:, :, 0]
full = cv2.imread("X:\\Pupil Dataset\\Dikablis_Full\\dikablisR_1_1_0_I.png")[:, :, 0]
minmax = procedure_min_max_simple_to_be_used(mg)
minmax2 = procedure_min_max_simple_to_be_used(full)
smim = filter_savitzky_golay_to_be_used(minmax)[0:2]
temp2 = np.expand_dims(patch_tr, axis=-1).transpose((2, 0, 1))
temp2 = torch.tensor(temp2.astype(np.float32) / 255.).unsqueeze(0)

@timeit
def analyse_subcomponents():

    for a in range(1000):
        minmax2 = procedure_min_max_simple_to_be_used(mg)
        smim2 = filter_savitzky_golay_to_be_used(minmax)[0:2]
        edges = get_sobel_edge(patch_mg)
        ellipse = get_ellipse_parameters(patch_tr)
        score = get_auto_operaton()
        entropy, intensity = get_entropy_and_intensity(patch_tr, (50, 50), (50, 50), 90)
        mtr = convert_tensor_to_matrx()
        tsr = convert_matrx_to_tensor()

analyse_subcomponents()
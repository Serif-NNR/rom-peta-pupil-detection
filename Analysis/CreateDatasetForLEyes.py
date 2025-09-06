import glob, cv2
import numpy as np
import matplotlib.pyplot as plt

from LEyesEvaluate import dataset_loaders

lpw_dir = glob.glob("P:\\LPW\\LPW_FULL\\*_I.png")
leyes_dir = "P:\\LPW\\LPW_LEyes\\"

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

def get_cropped_version(im, center):
    im, _, ox, oy = dataset_loaders.process_image(im, None, None, 128, crop_mode=2,
                                                   crop_center=(int(center[0] / 1), int(center[1] / 1)),
                                                   device="cpu:0")
    return (im.detach().cpu().numpy()[0, :, :] * 255).astype(np.uint8), int(ox), int(oy)


for lpw_im in lpw_dir:
    name = lpw_im.split("\\")[-1]
    image, mask = cv2.imread(lpw_im)[:, :, 0], cv2.imread(lpw_im.replace("_I.png", "_M.png"))[:, :, 0]
    ellipse = get_ellipse_parameters(mask)
    if ellipse[0] != (0,0):
        _, ox, oy = get_cropped_version(image, ellipse[0])
        mask = mask[oy:oy+128, ox:ox+128]
        image = image[oy:oy + 128, ox:ox + 128]
    cv2.imwrite(leyes_dir + name, image)
    cv2.imwrite(leyes_dir + name.replace("_I.png", "_M.png"), mask)

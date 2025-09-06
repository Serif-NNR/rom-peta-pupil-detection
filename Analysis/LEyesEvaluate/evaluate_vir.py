import cv2
import itertools
import copy, math
import os
import pandas as pd
import glob
import datetime
import torch
import torch.nn as nn
from torchvision.ops import focal_loss
import numpy as np
from einops.layers.torch import Reduce
from skimage.measure import regionprops, label
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from LEyesEvaluate.evaluate import read_file_as_dictionary


class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, h_c):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, h_c, kernel_size=1, padding="same")

        self.b2 = batchnorm_relu(h_c)
        self.c2 = nn.Conv2d(h_c, h_c, kernel_size=3, padding="same")

        self.b3 = batchnorm_relu(h_c)
        self.c3 = nn.Conv2d(h_c, in_c, kernel_size=1, padding="same")

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)

        x = self.b2(x)
        x = self.c2(x)

        x = self.b3(x)
        x = self.c3(x)

        skip = x + inputs
        return skip


class encoder_block(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.r = residual_block(in_c, int(in_c // 2))
        self.downsample = nn.MaxPool2d(2)

    def forward(self, inputs):
        skip = self.r(inputs)
        x = self.downsample(skip)

        return x, skip


class decoder_block(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c, int(in_c // 2))

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = x + skip
        x = self.r(x)
        return x


class build_resunet(nn.Module):
    """
    Hourglass model made of residual blocks based on the chinese CHI paper

    - the hourglass halves the feature map after each layer but keeps the depth the same (64)
    - the residual blocks consist of three convolutions. c(64, 32, 1x1) -> c(32, 32 3x3) -> c(32, 64, 1x1) -> add input to output
    """
    def __init__(self, conv_dims=64, base_dims=64, device="cpu"):
        super().__init__()
        self._device = device

        """ Input """
        self.c0 = nn.Conv2d(1, conv_dims, kernel_size=1)
        self.br0 = batchnorm_relu(conv_dims)

        """ Encoder  """
        self.e1 = encoder_block(conv_dims)
        self.e2 = encoder_block(conv_dims)
        self.e3 = encoder_block(conv_dims)
        self.e4 = encoder_block(conv_dims)
        self.e5 = encoder_block(conv_dims)
        self.e6 = encoder_block(conv_dims)

        """ Bridge """
        self.b1 = residual_block(base_dims, int(base_dims // 2))
        self.b2 = residual_block(base_dims, int(base_dims // 2))
        self.b3 = residual_block(base_dims, int(base_dims // 2))

        """ Decoder """
        self.d_1 = decoder_block(conv_dims)
        self.d0 = decoder_block(conv_dims)
        self.d1 = decoder_block(conv_dims)
        self.d2 = decoder_block(conv_dims)
        self.d3 = decoder_block(conv_dims)
        self.d4 = decoder_block(conv_dims)

        """ Output """

        # CR head
        self.ccr = nn.Conv2d(conv_dims, 5, kernel_size=1)
        self.bcr = nn.BatchNorm2d(5)

        # Attention
        self.max_pooling_layer = Reduce('b c h w -> b 1 h w', 'max')
        self.avg_pooling_layer = Reduce('b c h w -> b 1 h w', 'mean')
        self.ca = nn.Conv2d(2, 1, kernel_size=7, padding="same")

        # Pupil head
        self.cp = nn.Conv2d(conv_dims, 1, kernel_size=1)
        self.bp = nn.BatchNorm2d(1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax2d()

    def forward(self, inputs):
        """ Input """
        inp = self.c0(inputs)
        inp = self.br0(inp)

        """ Encoder"""
        x, skip1 = self.e1(inp)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)
        x, skip6 = self.e6(x)


        """ Bridge """
        b = self.b1(x)
        b = self.b2(b)
        b = self.b3(b)

        """ Decoder """
        d = self.d_1(b, skip6)
        d = self.d1(d, skip5)
        d = self.d1(d, skip4)
        d = self.d2(d, skip3)
        d = self.d3(d, skip2)
        d = self.d4(d, skip1)

        """ output """

        # CR head
        cr = self.ccr(d)
        cr = self.bcr(cr)

        # Attention
        max = self.max_pooling_layer(self.relu(cr))
        avg = self.avg_pooling_layer(self.relu(cr))

        attention = torch.cat([max, avg], 1)
        attention = self.ca(attention)
        attention = self.sigmoid(attention)

        # Pupil head
        cp = torch.multiply(attention, d)
        cp = self.cp(cp)
        cp = self.bp(cp)

        return cp, cr

    def predict(self, x, thres=0.9, centroid_method=None):
        """
        Plan:
        replace sigmoid with relu
        use argmax
        """
        predp, predc = self.forward(x.float().unsqueeze(1).to(self._device))
        predp = self.relu(predp)
        predc = self.relu(predc)

        predp = predp.squeeze(0)
        predc = predc.squeeze(0)

        if predp.dim() == 3:
            pred = torch.cat([predp, predc], 0).cpu().detach()
            pred = pred.unsqueeze(0)
        elif predp.dim() == 4:
            pred = torch.cat([predp, predc], 1).cpu().detach()
        else:
            raise Exception(
                f"Prediction dimension expected to be 3 or 4. Found (pupils){predp.dim()} and (crs){predc.dim()}")

        pred[pred < thres] = 0

        b, c, h, w = pred.shape
        pred = pred.reshape(-1, h, w)
        centroids = torch.vstack([(x==torch.max(x)).nonzero()[0].flip(-1) for x in pred])

        return centroids

    def refine(self, map, thres):
        map = self.relu(map)
        b, c, h, w = map.shape
        map = map.reshape(-1, h, w)

        peaks = torch.vstack([(x == torch.max(x)).nonzero()[0] for x in map]).detach().numpy()
        masks = self.sigmoid(map) > thres

        centroids = []
        for m, p in zip(masks, peaks):
            regions = regionprops(label(m))
            # find region whose centroid is closest to corresponding peak

            if len(regions) == 0:
                centroids.append(np.array([0, 0]))
            else:
                cs = np.array([r["Centroid"] for r in regions])
                closest_to_p = np.argmin(np.sqrt(np.sum((cs - p)**2, -1)))
                centroids.append(cs[closest_to_p][::-1])

        centroids = np.array(centroids)
        return centroids.reshape(b, c, 2)


    def predict_(self, x, thres=0.99, centroid_method=None):
        predp, predc = self.forward(x.float().unsqueeze(1).to(self._device))
        centroidp = self.refine(predp.cpu().detach(), thres)
        centroidc = self.refine(predc.cpu().detach(), thres)

        centroids = np.concatenate([centroidp, centroidc], axis=1)

        return centroids




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

ds = glob.glob("X:\\Pupil Dataset\\Dikablis_Full\\*_I.png")

def AnalyzeVirNet():
    model = torch.load("X:\\Fixein Pupil Detection\\LEyesEvaluate\\VIRNet.pt").to("cuda:0")
    ds_count, acc, err = 0, 0, 0
    previous_mask, previous_inference = None, None
    for dsi in ds:
        image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]
        image, mask = cv2.resize(image, (128, 128)), cv2.resize(mask, (128, 128))
        mask_ellipse = get_ellipse_parameters(mask)

        #im = np.expand_dims(image, axis=-1).transpose((2, 0, 1))
        im = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255

        p = model.predict(im.to("cuda:0"))
        p = p.nan_to_num(0)
        p = p.cpu().detach().numpy().squeeze()
        if False:
            out_cut = np.copy(p[0]["pmap"].detach().cpu().numpy())
            plt.imshow(out_cut)
            plt.show()
            print("stop")

        if np.isnan(p[0][0]) and (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): continue
        if np.isnan(p[0][0]): p = previous_inference
        if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0) and previous_mask != None: mask_ellipse = previous_mask

        px_err = math.dist(p[0], mask_ellipse[0])
        acc = acc + (1 if px_err <= 5 else 0)
        err = err + px_err

        if px_err > 5:
            print("["+str(ds_count)+"]",dsi, px_err, "   ", err)

        ds_count += 1
        previous_mask = mask_ellipse
        previous_inference = p

    print("\n\nds_count: {0}\n"
          "acc: {1}\n"
          "norm acc: {2}\n"
          "err: {3}\n"
          "norm err: {4}\n".format(
        ds_count,
        acc, acc / ds_count, err, err / ds_count
    ))







from LEyes_evaluate import do_hourglass_prediction_VirNet
def AnalyzeVirNetLikeLEyes():
    model = torch.load("X:\\Fixein Pupil Detection\\LEyesEvaluate\\VIRNet.pt").to("cuda:0")
    dictPure = read_file_as_dictionary("C:\\Users\\Sheriffnnr\\Desktop\\PD_AfterReview\\Banchmark\\filePure.txt")
    ds_count, acc, err = 0, 0, 0
    previous_mask, previous_inference = None, None
    for dsi in ds:
        image, mask = cv2.imread(dsi)[:, :, 0], cv2.imread(dsi.replace("_I.png", "_M.png"))[:, :, 0]


        p, confidence, offset, X, mask = do_hourglass_prediction_VirNet(model, "cuda:0", image, dictPure[dsi], 0.7, 128, 1, mask)
        mask_ellipse = get_ellipse_parameters(mask)
        p = p.nan_to_num(0)
        p = p.cpu().detach().numpy().squeeze()
        if False:
            out_cut = np.copy(p[0]["pmap"].detach().cpu().numpy())
            plt.imshow(out_cut)
            plt.show()
            print("stop")

        if np.isnan(p[0][0]) and (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0): continue
        if np.isnan(p[0][0]): p = previous_inference
        if (mask_ellipse[0][0] == 0 and mask_ellipse[0][1] == 0) and previous_mask != None: mask_ellipse = previous_mask

        px_err = math.dist(p[0], mask_ellipse[0])
        acc = acc + (1 if px_err <= 5 else 0)
        err = err + px_err

        if px_err > 5:
            print("[" + str(ds_count) + "]", dsi, px_err, "   ", err)

        ds_count += 1
        previous_mask = mask_ellipse
        previous_inference = p

    print("\n\nds_count: {0}\n"
          "acc: {1}\n"
          "norm acc: {2}\n"
          "err: {3}\n"
          "norm err: {4}\n".format(
        ds_count,
        acc, acc / ds_count, err, err / ds_count
    ))




AnalyzeVirNet()


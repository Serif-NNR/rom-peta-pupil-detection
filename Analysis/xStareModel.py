import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import skimage.exposure as exposure

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        #print("DoubleConv"+str(x.shape))
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def get_ellipse_parameters(map):
    map = (map.detach().cpu().numpy()[0, 0, :, :] * 255).astype(np.uint8)
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

def get_sobel_edge(map):
    sobelx = cv2.Sobel(map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(map, cv2.CV_64F, 0, 1, ksize=3)
    sobelx2 = cv2.multiply(sobelx, sobelx)
    sobely2 = cv2.multiply(sobely, sobely)
    sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    edge_im = exposure.rescale_intensity(sobel_magnitude, in_range='image', out_range=(0, 255)).clip(0, 255).astype(
        np.uint8)
    return edge_im

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

    #print(mask.shape, feature_map.shape, length)
    intensity = np.sum(intensity_map[mask > 0]) / total_pixel_count
    entropy = np.sum(enropy_map[mask > 0]) / total_pixel_count

    return entropy, intensity

class xStareROM(nn.Module):
    def __init__(self):
        super(xStareROM, self).__init__()
        self.name = "xStareROM_"

        self.fim_first = DoubleConv(1, 8)
        self.fim_last = DoubleConv(8, 1)

        self.pim = DoubleConv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 1)

        self.fdown1 = Down(1, 8)
        self.fdown2 = Down(8, 16)
        self.fdown3 = Down(16, 32)
        self.fdown4= Down(32, 64)
        self.fup4 = Up(96, 32)
        self.fup3 = Up(48,16)
        self.fup2 = Up(24, 8)
        self.fup1 = Up(9, 1)

        self.out = OutConv(1, 1)


    def fim_patches(self, x):
        x = x[:, :, :, 0:-2]
        _, _, height, width = x.shape
        slice_width, slice_height = width // 3, height // 2
        slices = [x[:, :, j * slice_height:(j + 1) * slice_height, i * slice_width:(i + 1) * slice_width] for i in range(3) for j in range(2)]
        return torch.cat(slices, dim=0)

    def fim_image(self, fim):
        im1, im2, im3 = torch.cat([fim[0, :, :, :], fim[1, :, :, :]], dim=1), torch.cat([fim[2, :, :, :], fim[3, :, :]], dim=1), torch.cat([fim[4, :, :, :], fim[5, :, :, :]], dim=1)
        im = torch.cat([im1, im2, im3], dim=2)
        im = F.pad(im, (0, 2, 0, 0))
        return im

    def get_scored_map(self, fim, ellipse, start_y, end_y, start_x, end_x):
        feature_map = (fim.detach().cpu().numpy()[0, :, :] * 255).astype(np.uint8)
        center = ((ellipse[0][0] * 3) + start_x, (ellipse[0][1] * 3) + start_y)
        size = (ellipse[1][0] * 3, ellipse[1][1] * 3)
        if center[0] < 0 or center[1] < 0 or size[0] < 5 or size[1] < 5: return 0.8, feature_map
        entropy, intensity = get_entropy_and_intensity(feature_map, center, size, ellipse[2])
        score = ((255 - entropy) / 255) * ((255 - intensity) / 255) / .8
        feature_map[int(center[1]):int(center[1]+size[1]), int(center[0]):int(center[0]+size[0])] = \
            feature_map[int(center[1]):int(center[1]+size[1]), int(center[0]):int(center[0]+size[0])] * score
        return score, feature_map

    def encoder_decoder(self, x):
        x1 = self.fdown1(x)
        x2 = self.fdown2(x1)
        x3 = self.fdown3(x2)
        x4 = self.fdown4(x3)
        x0 = self.fup4(x4, x3)
        x0 = self.fup3(x0, x2)
        x0 = self.fup2(x0, x1)
        x0 = self.fup1(x0, x)
        return self.out(x0)

    def shape_equalizer(self, x, result):
        diffY = x.size()[2] - result.size()[1]
        diffX = x.size()[3] - result.size()[2]
        result = F.pad(result, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        return result

    def forward(self, x, start_y, end_y, start_x, end_x):
        fim = self.fim_patches(x)
        fim = self.fim_last(self.fim_first(fim))
        fim = self.fim_image(fim)
        if fim.shape != x.shape: fim = self.shape_equalizer(x, fim)
        ellipse = [[0, 0]]
        pim = x[:, :, start_y:end_y, start_x:end_x]
        if pim.size()[2] > 10 and pim.size()[3] > 10:
            pim_d0 = self.pim(pim)
            pim_d1 = self.down1(pim_d0)
            pim_d2 = self.down2(pim_d1)
            pim_d3 = self.down3(pim_d2)
            ellipse = get_ellipse_parameters(pim_d3)

        if ellipse[0] != [0, 0]:
            score, map = self.get_scored_map(fim, ellipse, start_y, end_y, start_x, end_x)
            map = torch.tensor(np.expand_dims(map, axis=-1).transpose((2, 0, 1)).astype(np.float32) / 255.).unsqueeze(0).to("cuda:0")
            result = self.encoder_decoder(map)
        else:
            map = fim.unsqueeze(0)
            result = self.encoder_decoder(map)

        return result








if __name__ == "__main__":
    model = xStareROM()
    print(model(torch.randn((1, 1, 240, 320)), (50, 100), (100, 200)))
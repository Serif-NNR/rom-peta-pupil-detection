import os, cv2
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch import cuda

from LEyesEvaluate import dataset_loaders

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def dice_coef_metric(pred, label):
    intersection = 2.0 * (pred * label).sum()
    union = pred.sum() + label.sum()
    if pred.sum() == 0 and label.sum() == 0:
        return 1.
    return intersection / union

def dice_coef_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice_loss = dice_coef_loss(pred, label)
    bce_loss = nn.BCELoss()(pred, label)
    return dice_loss + bce_loss

def compute_iou(model, loader, threshold=0.3):
    valloss = 0
    with torch.no_grad():
        for step, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)

            outputs = model(data)#["out"]
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            loss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += loss
    return valloss / step

import gc
def train_model(model, train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs):
    loss_history = []
    train_history = []
    val_history = []
    dirName = "X:\\Fixein Pupil Detection\\ModelParameters\\" + str(time.time()).replace('.', '' + "_" + str(model.name))
    os.mkdir(dirName)
    for epoch in range(num_epochs):
        model.train()

        losses = []
        train_iou = []
        times = []

        for i, (image, mask, path) in enumerate(tqdm(train_loader)):
            # temp_time = datetime.now()
            image = image.to(device)
            mask = mask.to(device)
            # print(image.shape, mask.shape)
            #print("TRAIN: " + str(image.shape))
            outputs = model(image)#["out"]

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            # times.append((datetime.now()-temp_time).total_seconds())
            #cuda.empty_cache()
            if i % 100 == 0 and False:
                print(torch.cuda.memory_summary(device=None, abbreviated=False))

            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            #outputs = Variable(outputs.data, requires_grad=True)
            #mask = Variable(mask.data, requires_grad=True)
            loss = loss_func(outputs, mask)
            #ellipseFitError = EllipseFitErrorUNet(out_cut, mask) * 0.1
            #print(ellipseFitError)


            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # avg_ms = np.average(times)
        # print("AVG MS: " , avg_ms, "  FPS: ", 1/avg_ms)


        val_mean_iou = compute_iou(model, val_loader)
        scheduler.step(val_mean_iou)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)

        print('Epoch : {}/{}'.format(epoch + 1, num_epochs))
        print('loss: {:.3f} - dice_coef: {:.3f} - val_dice_coef: {:.3f}'.format(np.array(losses).mean(),
                                                                                np.array(train_iou).mean(),
                                                                                val_mean_iou))

        ''''''
        torch.save(model, dirName+"\\FPD_{0}_l{1:.3f}_d{2:.3f}_vd{3:.3f}.pt".format(epoch, np.array(losses).mean(),
                                                                                 np.array(train_iou).mean(),
                                                                                val_mean_iou))

    return loss_history, train_history, val_history




class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        #target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
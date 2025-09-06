import numpy as np
import torch
import time

from DynamicUnet import UNetDynamic
from ENet import ENet
from FlexiVit import FlexiVisionTransformer
from model_ASPP_keras import build_model_ASPP_keras
from model_CCSDG import UNetCCSDG, UNetCCSDG_S
from model_EgeUNet import EGEUNet
from model_TEC_CiT_Net import CIT
from oursegmodel import SRO_NetNew, SRO_UNet
from segmodel import UNet, MyUNet, MyNet, DynamicNet, DynamicNet2, UNet_L2, SegNet_ResNet_Seq1, SegResNet, SegNet, \
    SegNet_VGG16_L2, UNetS
from torch_UnetPP import NestedUNet, NestedUNetS
from train_and_metric_new_model import train_model, compute_iou, bce_dice_loss, device, DiceLoss ############### NEW MODELLLLLLLLLLLLLLLLLLLL
from dataset_operation import DatasetAttributes
from TransUNet import VisionTransformer, get_r50_b16_config
from paddle_PSPNet_ResNet import PSPNet
from paddle_DeepLabv3_ResNet import DeepLabV3PResnet50
#from paddle_SegNext import SegNeXt, MSCAN, MSCAN_S
from torch_SegNext import SegNext
from paddle_PPMobileSeg import *
#from torch_ConvNext import *
from torch_DeepLabv3 import *
from torch_UnetPP import NestedUNet
from TransUNet import VisionTransformer, get_r50_b16_config, get_r50_b16_configS
from torch_DeepLabv3 import *
from LEyesEvaluate.models import PupilNet
from xStareModel import xStareROM


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    #print("OK")



#model = UNetS(1, 1).to(device)
model = xStareROM().to(device)
#model = DeepLabV3PResnet50().to(device)
#model = SegNeXt(MSCAN(), MSCAN_S(), 1)
#model = SegNext(num_classes=1, in_channnels=1)
#model = PPMobileSeg(1, MobileSeg_Tiny(device)).to(device)
#model = NestedUNet().to(device)
#model = SegNet_VGG16_L2().to(device)
#model = VisionTransformer(get_r50_b16_configS(), img_size=256).to(device)
#model = deeplabv3_mobilenet_v3_large().to(device)
#model = PPMobileSeg(1, MobileSeg_Tiny(device)).to(device)
#model = deeplabv3_mobilenet_v3_large(num_classes=1).to(device)
#model = NestedUNet(input_channels=1, num_classes=1).to(device)
#model = VisionTransformer(get_r50_b16_config(), img_size=256).to(device)
#model = FlexiVisionTransformer().to(device)
#model = NestedUNetS().to(device)

#print(sum([np.prod(p.size()) for p in model_parameters]))




if __name__ == '__main__':
    from endpoint_load_save_model import State
    #model = NestedUNetS().to(device)

    #model = torch.load("X:\\Fixein Pupil Detection\\ModelParameters\\1695223462_torch_UNetPPS_AV_2005806\\FPD_6_l0.018_d0.963_vd0.902.pt")

    #State.checkpoint("xstare_nesteds.pth.tar", model)

    #model = PupilNet().to(device)  # for LEyes

    '''model2 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model3 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model4 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model5 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model6 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model7 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model8 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model9 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model10 = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1689522306_UNet_BAUG_FULL7804434\\FPD_0_l0.053_d0.928_vd0.756.pt")
    model = torch.load(
        "X:\\Fixein Pupil Detection\\ModelParameters\\1691069852_torch_UNetPPS_x4_x4_4377315\\FPD_2_l0.027_d0.950_vd0.884.pt")
    '''

    #model = ENet().to("cuda:0")
    #model = EGEUNet(input_channels=1).to(device)
    #model = CIT(window_size=2).to(device)
    #model = build_model_ASPP_keras((240,320, 1))
    #model = UNetCCSDG_S().to(device)
    #model = PSPNet(device=device).to(device)
    #model = NestedUNetS().to(device)

    import sys
    print(sys.getsizeof(model))
    #out = model(torch.randn(1, 1, 256, 256).to(device))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print("MODEL NAME :  " + str(model))
    print("MODEL PARM :  " + str(params))

    #model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    dataset = DatasetAttributes("P:\\LPW\\LPW_FULL\\", new_ds=True, complex_ds=False) # USE comples=True for LPW-AV
    #dataset = DatasetAttributes("P:\\LPW\\LPW_LEyes\\", new_ds=True)            # for LEyes
    #dataset = DatasetAttributes("P:\\LPW\\LPW_BAUG\\", new_ds=True)
    #dataset.Shortly(RATE=41)
    dataset.Shortly(RATE=1)
    #datasetValid = DatasetAttributes("P:\\LPW\\LPW_FULL\\", new_ds=True)
    # dataset.Shortly(RATE=41)
    #datasetValid.Shortly(RATE=1)




    #print(dataset.dirName)
    force_cudnn_initialization()
    #print(torch.cuda.is_available())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    num_epochs = 100
    #loss_history, train_history, val_history = train_model(model, dataset.train_dataloader, dataset.val_dataloader, bce_dice_loss, optimizer, scheduler, num_epochs)

    loss = DiceLoss(1)
    loss_history, train_history, val_history = train_model(model, dataset.train_dataloader, dataset.val_dataloader, loss, optimizer, scheduler, num_epochs)





    test_iou = compute_iou(model, dataset.test_dataloader)
    print("TEST Mean IoU: {:.3f}%".format(100*test_iou))

    val_iou = compute_iou(model, datasetValid.val_dataloader)
    print("VALD Mean IoU: {:.3f}%".format(100 * val_iou))

















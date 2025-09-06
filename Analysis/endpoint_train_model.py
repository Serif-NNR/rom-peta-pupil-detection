import numpy as np
import torch
import time
from segmodel import UNet, MyUNet, MyNet, DynamicNet, DynamicNet2, UNet_L2
from train_and_metric import train_model, compute_iou, bce_dice_loss, device
from dataset_operation import DatasetAttributes
from LEyesEvaluate.models import PupilNet

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    #print("OK")



#model = UNet_L2(1, 1).to(device)
model = PupilNet().to(device) # for LEyes
out = model(torch.randn(1, 1, 256, 256).to(device))

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#print(sum([np.prod(p.size()) for p in model_parameters]))

#dataset = DatasetAttributes("P:\\LPW\\FULL_NEW\\", new_ds=True)
dataset = DatasetAttributes("P:\\LPW\\LPW_LEyes\\", new_ds=True)
dataset.Shortly()


if __name__ == '__main__':
    #print(dataset.dirName)
    force_cudnn_initialization()
    #print(torch.cuda.is_available())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    num_epochs = 100
    loss_history, train_history, val_history = train_model(model, dataset.train_dataloader, dataset.val_dataloader, bce_dice_loss, optimizer, scheduler, num_epochs)




    test_iou = compute_iou(model, dataset.test_dataloader)
    print("TEST Mean IoU: {:.3f}%".format(100*test_iou))

    val_iou = compute_iou(model, dataset.val_dataloader)
    print("VALD Mean IoU: {:.3f}%".format(100 * val_iou))


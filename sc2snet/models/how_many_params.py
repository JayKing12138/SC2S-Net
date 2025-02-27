import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from tractseg.models.unet_pytorch_deepsup import UNet_Pytorch_DeepSup


torch.cuda.set_device(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
model = UNet_Pytorch_DeepSup().to(device)

summary(model, (1, 28, 28))



# # 查看网络params占用
# from torchstat import stat
# from tractseg.models.unet_pytorch_deepsup import UNet_Pytorch_DeepSup
 
# model = UNet_Pytorch_DeepSup()
# stat(model, (3, 224, 224))
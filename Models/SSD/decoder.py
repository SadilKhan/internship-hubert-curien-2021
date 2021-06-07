import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
#from torchvision.ops import RoIAlign
from torchsummary import summary

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    # ROIALIGN
    #self.roialign=RoIAlign((2816,19,19),1,-1)

    # Decoding Layers
    self.convT_1=nn.ConvTranspose2d(2816,256,kernel_size=3,stride=2) #(256,39,39)
    self.convT_2=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2) #(128,79,79)
    self.upsample_1=nn.Upsample((80,80)) #(128,80,80)

    self.convT_3=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2)
    self.upsample_2=nn.Upsample((160,160))

    self.convT_4=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2)
    self.convT_5=nn.ConvTranspose2d(32,3,kernel_size=1,stride=2)
    self.upsample_3=nn.Upsample((640,640))

  def forward(self,x):
    x=x.cpu()
    out=self.convT_1(x)
    out=self.convT_2(out)
    out=self.upsample_1(out)
    out=self.convT_3(out)
    out=self.upsample_2(out)
    out=self.convT_4(out)
    out=self.convT_5(out)
    out=self.upsample_3(out)
    return out


if __name__=="__main__":
    decoder=Decoder()
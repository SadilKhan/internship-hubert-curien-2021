import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from SSD.box_utils import cxcy_to_gcxgcy,cxcy_to_xy,xy_to_cxcy,gcxgcy_to_cxcy,find_jaccard_overlap

class Decoder(nn.Module):
  def __init__(self):
    super(Decoder,self).__init__()
    """ 
    Parameter:
    detector --> Object Detector
    output_size --> Output size of ROI Align step. Default (5,5)
    """
    # Decoding Layers
    self.convT_1=nn.ConvTranspose2d(2560,1024,kernel_size=3,stride=2) #(1024, 11, 11)
    self.convT_2=nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2) #(512, 23, 23)
    self.convT_3=nn.ConvTranspose2d(512,256,kernel_size=3,stride=1) #(256, 25, 25)

    self.convT_4=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2) # (128, 51, 51)
    self.convT_5=nn.ConvTranspose2d(128,3,kernel_size=3,stride=2) # (3, 103, 103)
    self.upsample_1=nn.Upsample((100,100)) # (3, 100, 100)


  def forward(self,x):
    # Output Images for batches
    self.__batch_size=len(x)
    output=[]
    for i in range(self.__batch_size):
      out=F.relu(self.convT_1(x[i]))
      out=F.relu(self.convT_2(out))
      out=F.relu(self.convT_3(out))
      out=F.relu(self.convT_4(out))
      out=F.relu(self.convT_5(out))
      #out=F.relu(self.convT_6(out))
      #out=F.relu(self.convT_7(out))
      out=self.upsample_1(out)
      output.append(out)

    return output


if __name__=="__main__":
    decoder=Decoder()

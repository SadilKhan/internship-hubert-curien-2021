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
    self.decoder=nn.Sequential(
        nn.ConvTranspose2d(1024,256,kernel_size=5,stride=2,output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(256,32,kernel_size=3,stride=2,output_padding=1),
        nn.ReLU(),
        nn.Upsample((32,32)),
        nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16,3,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.Sigmoid()
    )

  def forward(self,x):
    # Output Images for batches
    self.__batch_size=len(x)
    output=[]
    for i in range(self.__batch_size):
      out=self.decoder(x[i])
      output.append(out)

    return output


if __name__=="__main__":
    decoder=Decoder()

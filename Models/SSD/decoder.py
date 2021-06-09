import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
#from torchvision.ops import RoIAlign
from torchsummary import summary
from box_utils import cxcy_to_gcxgcy,cxcy_to_xy,xy_to_cxcy,gcxgcy_to_cxcy,find_jaccard_overlap

class Decoder(nn.Module):
  def __init__(self,detector=None,threhsold=0.75):
    super(Decoder,self).__init__()
    # The feature maps in detector will work as encoded information.
    self.detector=detector
        
    
    # ROIALIGN
    self.roialign=RoIAlign((2816,3,3),1,-1)

    # Decoding Layers
    self.convT_1=nn.ConvTranspose2d(2816,256,kernel_size=3,stride=2) #(256,39,39)
    self.convT_2=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2) #(128,79,79)
    self.upsample_1=nn.Upsample((80,80)) #(128,80,80)

    self.convT_3=nn.ConvTranspose2d(128,64,kernel_size=3,stride=2)
    self.upsample_2=nn.Upsample((160,160))

    self.convT_4=nn.ConvTranspose2d(64,32,kernel_size=3,stride=2)
    self.convT_5=nn.ConvTranspose2d(32,3,kernel_size=1,stride=2)
    self.upsample_3=nn.Upsample((640,640))

  def forward(self,mask):
    x,self.boxes=mask

    # Get RoI's
    self.locs,_=cxcy_to_xy(gcxgcy_to_cxcy(self.detector(x),self.detector.priors_cxcy))



    # Get all the feature maps
    self.get_feature_maps(x)
    self.merge_feature_maps()

    out=self.convT_1(x)
    out=self.convT_2(out)
    out=self.upsample_1(out)
    out=self.convT_3(out)
    out=self.upsample_2(out)
    out=self.convT_4(out)
    out=self.convT_5(out)
    out=self.upsample_3(out)
    return out
  
  def get_feature_maps(self,x):
    """ Get all the feature maps from the ssd detector"""
    self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1)).cuda()  # there are 512 channels in conv4_3_feats
    

    # Get Conv4 and Conv7 from VGG
    self.conv4_3,self.conv7=self.detector.base(x)

    # Rescale conv4_3 after L2 norm
    norm = self.conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
    self.conv4_3_feats = self.conv4_3_feats / norm  # (N, 512, 38, 38)
    self.conv4_3_feats = self.conv4_3_feats * self.rescale_factors 

    # Get rest of the feature maps from
    self.conv8_2_feats, self.conv9_2_feats, self.conv10_2_feats, self.conv11_2_feats = self.aux_convs(self.conv7_feats)
    
  
  def merge_feature_maps(self):
    """ Merge all Fearure Maps"""
    # Feature Map ROI Postions in Locs
    self.__roi_pos={"conv4_3":[0,5776],"conv7":[5776,7942],
                 "conv8_2":[7942,8542],"conv9_2":[8542,8692],"conv10_2":[8692,8728],
                 "conv11_2":[8728,8732]}

    # Region of Interest
    self.__roi={"conv4_3":self.locs[:,:5776,:],"conv7":self.locs[:,5776:7942,:],"conv8_2":self.locs[:,7942:8542,:],
             "conv9_2":self.locs[:,8542:8692,:],"conv10_2":self.locs[:,8692:8728,:],"conv11_2":self.locs[:,8728:8732,:]}
    
    # Find Roi's for conv4_3, since we will use roialign here
    self.__roi_conv4_3=self.find_roi("conv4_3",(38,38))
  
  
  def find_roi(self,feature_name,feature_size,threhsold=0.75):
    """ It finds the RoI's for a specific feature name"""
    self.__bacth_size=self.locs.size(0)
    roi_sp=[]
    min_,max_=self.__roi_pos[feature_name]

    for i in range(self.__batch_size):
      n_objects=self.boxes[i].size(0)
      overlap=find_jaccard_overlap(self.boxes[i],torch.clamp(self.detector.priors_xy[min_:max_,:],min=0))

      _, prior_for_each_object = overlap.max(dim=1)
      locs_for_map=torch.clamp(self.locs[i][min_:max_,:][prior_for_each_object],min=0,max=1)*(feature_size-1)
      roi_sp.append(locs_for_map)
  
    return roi_sp

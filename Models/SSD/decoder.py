import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from torchsummary import summary
from SSD.box_utils import cxcy_to_gcxgcy,cxcy_to_xy,xy_to_cxcy,gcxgcy_to_cxcy,find_jaccard_overlap

class Decoder(nn.Module):
  def __init__(self,detector=None,output_size=(5,5)):
    super(Decoder,self).__init__()
    """ 
    Parameter:
    detector --> Object Detector
    output_size --> Output size of ROI Align step. Default (5,5)
    """

    # The feature maps in detector will work as encoded information.
    self.detector=detector
    self.output_size=output_size
        
     # ROIALIGN
    self.roialign=RoIAlign(self.output_size,1,-1)
    

    # Decoding Layers
    self.convT_1=nn.ConvTranspose2d(self.num_filter,1024,kernel_size=3,stride=2) #(1024,11,11)
    self.convT_2=nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2,padding=2) #(512,19,19)
    self.convT_3=nn.ConvTranspose2d(512,256,kernel_size=3,stride=2) #(256,39,39)

    self.convT_4=nn.ConvTranspose2d(256,128,kernel_size=3,stride=2) # (128,79,79)
    self.convT_5=nn.ConvTranspose2d(128,64,kernel_size=5,stride=2) # (64, 161, 161)

    self.convT_6=nn.ConvTranspose2d(64,32,kernel_size=1,stride=2) # (32, 321, 321)
    self.convT_7=nn.ConvTranspose2d(32,3,kernel_size=1,stride=2) # (3, 641, 641)
    self.upsample_1=nn.Upsample((640,640)) # (3,640,640)


  def forward(self,x,boxes=None):
    self.boxes=boxes

    # Get RoI's offset
    self.locs,self.classes=self.detector(x)
    self.classes=self.classes.cpu().detach()
    self.locs=self.locs.cpu().detach()

    # Transfer ssd style coordinates to xy coordinates
    self.__batch_size=self.locs.size(0)
    for i in range(self.__batch_size):
      self.locs[i]=cxcy_to_xy(gcxgcy_to_cxcy(self.locs[i],self.detector.priors_cxcy.cpu()))

    # Get all the feature maps
    self.feature_map=self.get_feature_maps(x)

    out=F.relu(self.convT_1(self.feature_map))
    out=F.relu(self.convT_2(out))
    out=F.relu(self.convT_3(out))
    out=F.relu(self.convT_4(out))
    out=F.relu(self.convT_5(out))
    out=F.relu(self.convT_6(out))
    out=F.relu(self.convT_7(out))
    out=self.upsample_1(out)
    return out
  
  def get_feature_maps(self,x):
    """ Get all the feature maps from the ssd detector"""
    
    self.rescale_factors = nn.Parameter(torch.cuda.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
    # ROI Postions in Locs data Dictinary
    self.__roi_pos_dict={"conv4_3":[0,38*38],"conv7":[38*38,38*38+19*19],
                 "conv8_2":[38*38+19*19,38*38+19*19+10*10],"conv9_2":[38*38+19*19+10*10,38*38+19*19+10*10+5*5],
                 "conv10_2":[38*38+19*19+10*10+5*5, 38*38+19*19+10*10+5*5+3*3]}
    self.feature_map_size={"conv4_3":38,"conv7":19,"conv8_2":10,"conv9_2":5,"conv10_2":3}
    self.__feat_map_dict={"conv4_3":self.conv4_3,"conv7":self.conv7,
                  "conv8_2":self.conv8_2,"conv9_2":self.conv9_2,"conv10_2":self.conv10_2,
                  "conv11_2":self.conv11_2}

    # Get Conv4 and Conv7 from VGG
    self.conv4_3,self.conv7=self.detector.base(x)

    # Rescale conv4_3 after L2 norm
    norm = self.conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
    self.conv4_3 = self.conv4_3 / norm  # (N, 512, 38, 38)
    self.conv4_3 = self.conv4_3 * self.rescale_factors 

    # Get rest of the feature maps from conv7
    self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2 = self.detector.aux_convs(self.conv7)


    # For every object get it respective locations in feature map
    self.locations=self.get_map_for_objects()

    # Merge feature maps 
    feature_map=self.merge_feature_maps()
    return feature_map
  
  def get_map_for_objects(self):
    # Get the locations of objects in feature maps
    locations=[]
    for i in range(self.__batch_size):
      sc_bx,cl_bx=self.classes[i].max(dim=1)
      un_cl=torch.unique(cl_bx)
      # The locations of objects in every feature maps
      class_dict={}
      for num,k in enumerate(un_cl):
        # We don't need background so eliminate object 0
        if k!=0:
          new_bx_k=[]
          bx_k=(cl_bx==k).nonzero(as_tuple=True)[0].cpu()
          for rp in list(self.__roi_pos_dict.keys()):
            min_,max_=self.__roi_pos_dict[rp]
            max_bx_k=bx_k[bx_k<max_]
            range_bx_k=max_bx_k[max_bx_k>=min_]
            # Extract the feature map name where positions belong
            try:
              locs_for_k=torch.clamp(self.locs[i][range_bx_k[torch.argmax(sc_bx[range_bx_k]).item()].item()],min=0,max=1)*self.feature_map_size[rp]
              new_bx_k+=[dict({rp:locs_for_k})]
            except:
              new_bx_k+=[]
          class_dict[k.cpu().item()]=new_bx_k
      locations.append(class_dict)
    return locations
    
  
  def merge_feature_maps(self):
    """ Merge the Fearure Maps for every object."""
    feature_map=[]
    upsample=nn.Upsample((2560,5))
    for i in range(self.__batch_size):
      # Feature Maps for every Batch. Separate feature maps for separate images.
      feature_map_bt=[]
      for ob in self.locations[i].keys():
        # Feature maps for every object. We will concatenate the feature maps
        temp_ob=[]
        for fm in self.locations[i][ob].keys():
          roi=self.locations[i][ob][fm]
          roi=[0]+roi
          roi=torch.cuda.FloatTensor(roi).unsqueeze(0)
          aligned_image=self.roiAlign(self.__feat_map_dict[fm],roi)
          temp_ob.append(aligned_image)
        
        temp_ob=torch.moveaxis(torch.cat(temp_ob,dim=1),1,2)
        temp_ob=upsample(temp_ob)
        temp_ob=torch.moveaxis(temp_ob,2,1)
        feature_map_bt.append(temp_ob)
      feature_map.append(torch.cat(feature_map_bt,dim=0))
    
    return feature_map
  


if __name__=="__main__":
    decoder=Decoder()

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
  def __init__(self,detector=None,output_size=(5,5),num_filter=50000):
    super(Decoder,self).__init__()
    """ 
    Parameter:
    detector --> Object Detector
    output_size --> Output size of ROI Align step. Default (5,5)
    filter --> Number of feature maps after merging all the feature maps
    """

    # The feature maps in detector will work as encoded information.
    self.detector=detector
    self.output_size=output_size
    self.num_filter=num_filter
        
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

    # Get Conv4 and Conv7 from VGG
    self.conv4_3,self.conv7=self.detector.base(x)

    # Rescale conv4_3 after L2 norm
    norm = self.conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
    self.conv4_3 = self.conv4_3 / norm  # (N, 512, 38, 38)
    self.conv4_3 = self.conv4_3 * self.rescale_factors 

    # Get rest of the feature maps from conv7
    self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2 = self.detector.aux_convs(self.conv7)


    # Merge all the Feature maps after roi aligning
    self.locations=self.get_map_for_objects()
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
        if k!=0:
          new_bx_k=[]
          bx_k=(cl_bx==k).nonzero(as_tuple=True)[0].cpu()
          for rp in list(self.__roi_pos_dict.keys()):
            min_,max_=self.__roi_pos_dict[rp]
            max_bx_k=bx_k[bx_k<max_]
            range_bx_k=max_bx_k[max_bx_k>=min_]
            try:
              locs_for_k=torch.clamp(self.locs[i][range_bx_k[torch.argmax(sc_bx[range_bx_k]).item()].item()],min=0,max=1)*self.feature_map_size[rp]
              new_bx_k+=[dict({rp:locs_for_k.tolist()})]
            except:
              new_bx_k+=[]
          class_dict[k.cpu().item()]=new_bx_k
      locations.append(class_dict)
    return locations
    
  
  def merge_feature_maps(self):
    """ Merge all Fearure Maps"""

    # Feature Maps Dictionary
    self.__feat_map_dict={"conv4_3":self.conv4_3,"conv7":self.conv7,
                 "conv8_2":self.conv8_2,"conv9_2":self.conv9_2,"conv10_2":self.conv10_2,
                 "conv11_2":self.conv11_2}

    self.feature_map=[]
    #assert self.__batch_size==len(locations)
    for i in range(self.__batch_size):
      for j in range(self.locations[i]):
        pass

    feature_map=torch.cat((self.__roi_conv4_3,self.__roi_conv7,self.__roi_conv8_2,self.__roi_conv9_2,self.__roi_conv10_2),dim=0)


    
    return feature_map
  
  def add_batch_number(self,boxes,batch):
    """ Add Batch Number in the first column of the rois """
    box_size=boxes.size(0)
    return torch.stack((torch.ones((box_size,1,4),device="cuda")*batch,boxes.reshape(box_size,1,-1)),dim=2).reshape(-1,8)[:,3:]
  
  
  def find_roi(self,feature_name,feature_size):
    """ It finds the RoI's for a specific feature name"""
    
    roi_sp=[]
    min_,max_=self.__roi_pos_dict[feature_name]
    

    for i in range(self.__batch_size):
      n_objects=self.boxes[i].size(0)
      overlap=find_jaccard_overlap(self.boxes[i],torch.clamp(self.detector.priors_xy[min_:max_,:],min=0))
      locs_xy=cxcy_to_xy(gcxgcy_to_cxcy(self.locs[i],self.detector.priors_cxcy))

      _, prior_for_each_object = overlap.max(dim=1)
      locs_for_map=torch.clamp(locs_xy[min_:max_,:][prior_for_each_object],min=0,max=1)*(feature_size[0]-1)
      locs_for_map=self.add_batch_number(locs_for_map,i)
      roi_sp.append(locs_for_map)
    roi_sp=torch.cat(roi_sp,dim=0)
    return roi_sp

    
  def roi_align(self,feature_name,feature_size):
    """ Roi Alignment """
    w,h=self.output_size
    
    self.__roi=self.find_roi(feature_name,feature_size)
    aligned_map=self.roialign(self.__feat_map_dict[feature_name],self.__roi)
    return aligned_map.reshape(self.__batch_size,-1,w,h)


if __name__=="__main__":
    decoder=Decoder()

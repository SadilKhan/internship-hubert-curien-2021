import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.ops import RoIAlign
from torchsummary import summary
from box_utils import cxcy_to_gcxgcy,cxcy_to_xy,xy_to_cxcy,gcxgcy_to_cxcy,find_jaccard_overlap

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
    self.convT_1=nn.ConvTranspose2d(self.num_filter,1024,kernel_size=3,stride=2) #(256,39,39)
    self.convT_2=nn.ConvTranspose2d(1024,512,kernel_size=3,stride=2) #(128,79,79)
    self.upsample_1=nn.Upsample((80,80)) #(128,80,80)

    self.convT_3=nn.ConvTranspose2d(512,128,kernel_size=3,stride=2)
    self.upsample_2=nn.Upsample((160,160))

    self.convT_4=nn.ConvTranspose2d(128,32,kernel_size=3,stride=2)
    self.convT_5=nn.ConvTranspose2d(32,3,kernel_size=1,stride=2)
    self.upsample_3=nn.Upsample((640,640))

  def forward(self,mask):
    x,self.boxes=mask

    # Get RoI's offset
    self.locs,_=self.detector(x)


    # Get all the feature maps
    x=self.get_feature_maps(x)

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
    self.rescale_factors = nn.Parameter(torch.cuda.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
    

    # Get Conv4 and Conv7 from VGG
    self.conv4_3,self.conv7=self.detector.base(x)

    # Rescale conv4_3 after L2 norm
    norm = self.conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
    self.conv4_3 = self.conv4_3 / norm  # (N, 512, 38, 38)
    self.conv4_3 = self.conv4_3 * self.rescale_factors 

    # Get rest of the feature maps from
    self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2 = self.detector.aux_convs(self.conv7)

    # Merge all the Feature maps after roi aligning
    feature_map=self.merge_feature_maps()
    return feature_map
    
  
  def merge_feature_maps(self):
    """ Merge all Fearure Maps"""
    # ROI Postions in Locs data Dictinary
    self.__roi_pos_dict={"conv4_3":[0,5776],"conv7":[5776,7942],
                 "conv8_2":[7942,8542],"conv9_2":[8542,8692],"conv10_2":[8692,8728],
                 "conv11_2":[8728,8732]}

    """# Region of Interest Dictionary
    self.__roi_dict={"conv4_3":self.locs[:,:5776,:],"conv7":self.locs[:,5776:7942,:],"conv8_2":self.locs[:,7942:8542,:],
             "conv9_2":self.locs[:,8542:8692,:],"conv10_2":self.locs[:,8692:8728,:],"conv11_2":self.locs[:,8728:8732,:]}"""

    # Feature Maps Dictionary
    self.__feat_map_dict={"conv4_3":self.conv4_3,"conv7":self.conv7,
                 "conv8_2":self.conv8_2,"conv9_2":self.conv9_2,"conv10_2":self.conv10_2,
                 "conv11_2":self.conv11_2}
    
    # Find Roi's for conv4_3, since we will use roialign here
    self.__roi_conv4_3=self.roi_align("conv4_3",(38,38))
    self.__roi_conv7=self.roi_align("conv7",(19,19))
    self.__roi_conv8_2=self.roi_align("conv8_2",(10,10))
    self.__roi_conv9_2=self.roi_align("conv9_2",(5,5))
    self.__roi_conv10_2=self.roi_align("conv10_2",(3,3))
    #self.__roi_conv11_2=self.roi_align("conv11_2",(3,3))

    feature_map=torch.cat((self.__roi_conv4_3,self.__roi_conv7,self.__roi_conv8_2,self.__roi_conv9_2,self.__roi_conv10_2),dim=1)

    # Since Feature maps dimensions are different for every image, we will take first 50000 filters
    if feature_map.size(1)>=self.num_filter:
      feature_map=feature_map[:,:self.num_filter,:,:]
    else:
      # We need to upsample
      upsample=nn.Upsample(5,self.num_filter)
      feature_map=torch.moveaxis(upsample(torch.moveaxis(feature_map,1,-1)),-1,1)
    assert feature_map.shape==torch.Size([self.__batch_size,self.num_filter,self.output_size[0],self.output_size[1]])

    
    return feature_map
  
  def add_batch_number(self,boxes,batch):
    """ Add Batch Number in the first column of the rois """
    box_size=boxes.size(0)
    return torch.stack((torch.ones((box_size,1,4),device="cuda")*batch,boxes.reshape(box_size,1,-1)),dim=2).reshape(-1,8)[:,3:]
  
  
  def find_roi(self,feature_name,feature_size):
    """ It finds the RoI's for a specific feature name"""
    self.__batch_size=self.locs.size(0)
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

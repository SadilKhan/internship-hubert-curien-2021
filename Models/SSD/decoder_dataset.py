import numpy as np # Linear Algebra
import gc
import pandas as pd # Data Processing, CSV file I/O (e.g. pd.read_csv)

import torch
from torchvision import models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision
from torchvision.ops import RoIAlign
from torchsummary import summary

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")
import timeit
from box_utils import cxcy_to_gcxgcy,cxcy_to_xy,gcxgcy_to_cxcy,xy_to_cxcy,find_jaccard_overlap,get_target_image




class DecoderDataset(Dataset):
    def __init__(self, detector,data,directory, is_test=False, transform=None,output_size=(5,5),ssd_type=4):
        #self.annotation_folder_path = csv_path
        self.detector=detector # Object Detector
        self.data=data # Contains the information about bounding boxes
        self.directory=directory # Contains the coordinate of the cropped images
        self.all_images=self.data['image_name'].unique()
        self.transform = transform
        self.is_test = is_test
        self.output_size=output_size
        self.ssd_type=ssd_type # 4box SSD or 1
        # ROIALIGN
        self.roialign=RoIAlign(self.output_size,1,-1)
        
    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        img = Image.open(self.directory+"/"+img_name)
        img = img.convert('RGB')
        
        annotations=self.data[self.data['image_name']==img_name]

        self.box = self.get_xy(annotations)

        self.new_box = self.box_resize(self.box, img)
        if self.transform is not None:
            img = self.transform(img)
        if not self.is_test:
            self.labels=torch.FloatTensor(annotations['label'].values).cuda()
            return img_name,img.cuda(), self.new_box, self.labels
        else:
            return img_name,img.cuda(),self.new_box,annotations["label"].values
    
    def __len__(self):
        return len(self.all_images)
        
    def get_xy(self, annotation):
        boxes=torch.cuda.FloatTensor(annotation[['xmin','ymin','xmax','ymax']].values)
        return boxes
        
    def box_resize(self, box, img, dims=(300, 300)):
        old_dims = torch.cuda.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        #old_dims=torch.FloatTensor([img.width, img.height, img.width, img.height]).unsqueeze(0)
        new_box = box.cuda() / old_dims
        new_dims = torch.cuda.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        #new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        #new_box = new_box * new_dims
        
        return new_box
    
    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        image_names=list()
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
          image_names.append(b[0])
          images.append(b[1])
          boxes.append(b[2])
          labels.append(b[3])

        images = torch.stack(images, dim=0)
        # Get RoI's offset
        self.locs,self.classes=self.detector(images)
        self.classes=self.classes.cpu().detach()
        self.classes=F.softmax(self.classes,dim=2)
        self.locs=self.locs.cpu().detach()

        # Transfer ssd style coordinates to xy coordinates
        self.__batch_size=self.locs.size(0)
        for i in range(self.__batch_size):
          self.locs[i]=torch.clamp(cxcy_to_xy(gcxgcy_to_cxcy(self.locs[i],self.detector.priors_cxcy.cpu())),min=0,max=1)

        # Get all the feature maps
        self.problem=0
        self.get_feature_maps(images) #(Batch Size, Num of Object, 5, 5)
        self.transform_locations_tolist()
        if self.is_test:
          return image_names,self.feature_map,self.locations,boxes,labels,self.classes
        else:
          return image_names, self.feature_map, self.locations,boxes  # tensor (N, 3, 300, 300),
    
    def get_feature_maps(self,x):
      """ Get all the feature maps from the ssd detector"""
      
      self.rescale_factors = nn.Parameter(torch.cuda.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
      # ROI Postions in Locs data Dictinary
      if self.ssd_type>1:
        self.__roi_pos_dict={"conv4_3":[0,38*38*4],"conv7":[38*38*4,(38*38*4+19*19*6)],
                    "conv8_2":[38*38*4+19*19*6,38*38*4+19*19*6+10*10*6],"conv9_2":[38*38*4+19*19*6+10*10*6,38*38*4+19*19*6+10*10*6+5*5*6,
                    ],"conv10_2":[8692,38*38*4+19*19*6+10*10*6+5*5*6+3*3*4],"conv11_2":[8728,8733]}
      else:
        self.__roi_pos_dict={"conv4_3":[0,38*38],"conv7":[38*38,38*38+19*19],
                  "conv8_2":[38*38+19*19,38*38+19*19+10*10],"conv9_2":[38*38+19*19+10*10,38*38+19*19+10*10+5*5]}
                          
      self.feature_map_size={"conv4_3":38,"conv7":19,"conv8_2":10,"conv9_2":5,"conv10_2":3,"conv11_2":1}

      # Get Conv4 and Conv7 from VGG
      self.conv4_3,self.conv7=self.detector.base(x)



      # Get rest of the feature maps from conv7
      self.conv8_2, self.conv9_2, self.conv10_2, self.conv11_2 = self.detector.aux_convs(self.conv7)

      self.__feat_map_dict={"conv4_3":self.conv4_3,"conv7":self.conv7,
                    "conv8_2":self.conv8_2,"conv9_2":self.conv9_2,"conv10_2":self.conv10_2,
                    "conv11_2":self.conv11_2}


      # For every object get it respective locations in feature map
      self.locations,self.scores=self.get_map_for_objects()

      # Merge feature maps 
      self.feature_map=self.merge_feature_maps()
    
    def get_map_for_objects(self):
      # Get the locations of objects in feature maps
      locations=[]
      scores=[]
      self.used_locs=[]
      for i in range(self.__batch_size):
        if not self.is_test:
          # Score per box and class per box
          sc_bx,cl_bx=self.classes[i].max(dim=1)
        else:
          # For Test dataset, we will consider that a box has an object if the confidence score is more than 0.001.
          ob_sc,ob_cl=self.classes[i][:,1:].max(dim=1)
          ob_b_sc=ob_sc>0.001
          sc_bx,cl_bx=[],[]
          for j in range(len(ob_b_sc)):
            if ob_b_sc[j]:
              sc_bx.append(ob_sc[j])
              cl_bx.append(ob_cl[j])
            else:
              sc_bx.append(1)
              cl_bx.append(0)
          sc_bx,cl_bx=torch.tensor(sc_bx),torch.tensor(cl_bx)

        # Unique Classes
        un_cl=torch.unique(cl_bx)
        # The locations of objects in every feature maps
        class_dict={}
        # The scores for the locations
        score_dict={}
        for num,k in enumerate(un_cl):
          # We don't need background so eliminate object 0
          if k!=0:
            new_bx_k=dict()
            score_for_k=dict()
            bx_k=(cl_bx==k).nonzero(as_tuple=True)[0].cpu()
            bx_k=bx_k[0].item()
            if cl_bx[bx_k]>=0:
              #print(bx_k)
              #print(f"Batch : {i}, class {k}, Number {len(bx_k)}")
              for rp in list(self.__roi_pos_dict.keys()):
                min_,max_=self.__roi_pos_dict[rp]
                # Extract the feature map name where positions belong
                if bx_k>=min_ and bx_k<=max_:
                  locs_for_k_rp=self.locs[i][bx_k]
                  if self.check_nms(locs_for_k_rp):
                    #print(locs_for_k_rp)
                    new_bx_k[rp]=torch.clamp(locs_for_k_rp,min=0,max=1)
                    score_for_k[rp]=sc_bx[bx_k].tolist()
              if len(list(new_bx_k.keys()))>0:
                class_dict[k.cpu().item()]=new_bx_k
                score_dict[k.cpu().item()]=score_for_k
        locations.append(class_dict)
        scores.append(score_dict)
      return locations,scores
    
    def check_nms(self,present_loc):
      eligible=True
      temp=self.used_locs
      if self.used_locs==[]:
        self.used_locs.append(present_loc)
        return eligible
      for i in range(len(self.used_locs)):
        if find_jaccard_overlap(self.used_locs[i].unsqueeze(0),present_loc.unsqueeze(0))[0][0].item()>0.5:
          eligible=False
          break
        else:
          temp.append(present_loc)
      self.used_locs=temp
      return eligible
      
    
    def merge_feature_maps(self):
      """ Merge the Fearure Maps for every object."""
      feature_map=[]
      upsample_roi=nn.Upsample((5,5))
      upsample=nn.Upsample((1024,self.output_size[-1]))
      for i in range(self.__batch_size):
        # Feature Maps for every Batch. Separate feature maps for separate images.
        feature_map_bt=[]
        for ob in self.locations[i].keys():
          # Feature maps for every object. We will concatenate the feature maps
          temp_ob=[]
          for fm in self.locations[i][ob].keys():
            roi=self.locations[i][ob][fm]*(self.feature_map_size[fm]-1)
            roi=roi.tolist()
            roi=[0]+roi
            roi=torch.round(torch.cuda.FloatTensor(roi))
            roi=roi.unsqueeze(dim=0)
            aligned_image=self.roialign(self.__feat_map_dict[fm],roi)
            sum_=torch.sum(torch.isnan(aligned_image))
            if sum_.item()>0:
              print(roi,fm,torch.sum(torch.isnan(self.__feat_map_dict[fm])))
          
            if aligned_image.size(1)<1024:
              aligned_image=torch.moveaxis(aligned_image,1,2)
              aligned_image=upsample(aligned_image)
              aligned_image=torch.moveaxis(aligned_image,2,1)
            temp_ob.append(aligned_image.detach())
            
          temp_ob=torch.cat(temp_ob,dim=0)
          feature_map_bt.append(temp_ob)
        feature_map.append(torch.cat(feature_map_bt,dim=0))
      
      return feature_map
    
    def transform_locations_tolist(self):
      """ Transform the dictionary to lists of boxes"""
      for i in range(self.__batch_size):
        all_boxes=[]
        for ob in self.locations[i].keys():
          boxes=list(self.locations[i][ob].values())
          all_boxes.append(boxes[0])
        self.locations[i]=torch.stack(all_boxes)

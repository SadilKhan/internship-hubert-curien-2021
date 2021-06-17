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
    def __init__(self, detector,data, imageData, imageArray, is_test=False, transform=None,output_size=(5,5)):
        #self.annotation_folder_path = csv_path
        self.detector=detector # Object Detector
        self.data=data # Contains the information about bounding boxes
        self.imageData=imageData # Contains the coordinate of the cropped images
        self.imageArray=imageArray # Contains the arrays of the original 18 images
        self.all_images=self.data['image_name'].unique()
        self.transform = transform
        self.is_test = is_test
        self.output_size=output_size
        # ROIALIGN
        self.roialign=RoIAlign(self.output_size,1,-1)
        
    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        if "_" in img_name:
          original_img_name=img_name.split("_")[0]+".jpg"
        else:
          original_img_name=img_name
        coord=self.imageData[self.imageData['image_name']==img_name][["x_min","y_min","x_max","y_max"]].values[0]
        img = Image.fromarray(self.imageArray[self.imageArray['image_name']==original_img_name]['image_array'].values[0][
            int(coord[1]):int(coord[3]),int(coord[0]):int(coord[2]),:])
        img = img.convert('RGB')
        
        if not self.is_test:
            annotations=self.data[self.data['image_name']==img_name]

            self.box = self.get_xy(annotations)

            self.new_box = torch.cuda.FloatTensor(self.box_resize(self.box, img))
            if self.transform:
                img = self.transform(img)
            

            #self.labels=torch.FloatTensor(annotations['label'].values).cuda()

            """# Encode the labels with Int
            self.le=LabelEncoder()
            self.labels=torch.FloatTensor(self.le.fit_transform(self.labels))"""

            return img_name,img.cuda(), self.new_box
            #return img_name,img,self.new_box
        else:
            return img_name,img.cuda()
    
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
        image_names=list() # Name of the images
        images = list()
        boxes = list()
        labels = list()
#         difficulties = list()

        for b in batch:
            image_names.append(b[0])
            images.append(b[1])
            boxes.append(b[2])
            #labels.append(b[3])
#             difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        # Get RoI's offset
        self.locs,self.classes=self.detector(images)
        self.classes=self.classes.cpu().detach()
        self.locs=self.locs.cpu().detach()

        # Transfer ssd style coordinates to xy coordinates
        self.__batch_size=self.locs.size(0)
        for i in range(self.__batch_size):
          self.locs[i]=cxcy_to_xy(gcxgcy_to_cxcy(self.locs[i],self.detector.priors_cxcy.cpu()))

        # Get all the feature maps
        self.get_feature_maps(images) #(Batch Size, Num of Object, 5, 5)
        self.make_single_box()

        return image_names,self.feature_map, self.locations,boxes  # tensor (N, 3, 300, 300),
    
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
      for i in range(self.__batch_size):
        sc_bx,cl_bx=self.classes[i].max(dim=1)
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
            for rp in list(self.__roi_pos_dict.keys()):
              min_,max_=self.__roi_pos_dict[rp]
              max_bx_k=bx_k[bx_k<max_]
              range_bx_k=max_bx_k[max_bx_k>=min_]
              # Extract the feature m0ap name where positions belong
              try:
                locs_for_k=torch.clamp(self.locs[i][range_bx_k[torch.argmax(sc_bx[range_bx_k]).item()].item()],min=0,max=1)*self.feature_map_size[rp]
                score_for_k[rp]=torch.max(sc_bx[range_bx_k]).item()
                new_bx_k[rp]=locs_for_k.tolist()
              except:
                pass
            class_dict[k.cpu().item()]=new_bx_k
            score_dict[k.cpu().item()]=score_for_k
        locations.append(class_dict)
        scores.append(score_dict)
      return locations,scores
      
    
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
            aligned_image=self.roialign(self.__feat_map_dict[fm],roi)
            temp_ob.append(aligned_image)
          
          temp_ob=torch.moveaxis(torch.cat(temp_ob,dim=1),1,2)
          temp_ob=upsample(temp_ob)
          temp_ob=torch.moveaxis(temp_ob,2,1)
          feature_map_bt.append(temp_ob)
        feature_map.append(torch.cat(feature_map_bt,dim=0))
      
      return feature_map
    
    def make_single_box(self):
      """ Some images have more than one boxes as they are captured in more than one feature maps. 
      We need to take the one with maximum scores"""
      for i,lc in enumerate(self.locations):
        for j,ob in enumerate(lc.keys()):
          
          if len(list(lc[ob].keys()))>1:
            represented_box=np.argmax(list(self.scores[i][ob].values()))
            self.scores[i][ob]=np.max(list(self.scores[i][ob].values()))
            lc[ob]=torch.tensor(lc[ob][list(lc[ob].keys())[represented_box]]).to("cuda")/self.feature_map_size[list(lc[ob].keys())[represented_box]]
          else:
            self.scores[i][ob]=list(self.scores[i][ob].values())[0]
            lc[ob]=torch.tensor(lc[ob][list(lc[ob].keys())[0]]).to("cuda")/self.feature_map_size[list(lc[ob].keys())[0]]
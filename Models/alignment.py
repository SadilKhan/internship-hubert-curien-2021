
import PIL.Image as Image
from PIL import ImageOps
import os 
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
import warnings
import torch.nn.functional as F
import pickle 
import pandas as pd
import kornia.geometry as tgm
#from scipy.misc import imresize
from itertools import product
import matplotlib.pyplot as plt 

## composite image    
def get_Avg_Image(Is, It) : 
    
    Is_arr, It_arr = np.array(Is) , np.array(It)
    Imean = Is_arr * 0.5 + It_arr * 0.5
    return Image.fromarray(Imean.astype(np.uint8))


def alignment(source,target,folder):
    import sys
    sys.path.append(folder+"/quick_start")
    sys.path.append(folder+'/model/')
    sys.path.append(folder+'/utils/')
    from coarseAlignFeatMatch import CoarseAlign
    
    import outil
    
    import model as model
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    resumePth = folder+'/model/pretrained/MegaDepth_Theta1_Eta001_Grad0_0.807.pth' ## model for visualization
    kernelSize = 7

    Transform = outil.Homography
    nbPoint = 4
        

    ## Loading model
    # Define Networks
    network = {'netFeatCoarse' : model.FeatureExtractor(), 
            'netCorr'       : model.CorrNeigh(kernelSize),
            'netFlowCoarse' : model.NetFlowCoarse(kernelSize), 
            'netMatch'      : model.NetMatchability(kernelSize),
            }
        

    for key in list(network.keys()) : 
        #network[key].cuda()
        typeData = torch.FloatTensor

    # loading Network 
    param = torch.load(resumePth,map_location=torch.device("cpu"))
    msg = 'Loading pretrained model from {}'.format(resumePth)
    print (msg)

    for key in list(param.keys()) : 
        network[key].load_state_dict(param[key] ) 
        network[key].eval()

    I1 = Image.open(target).convert('RGB')
    I2 = Image.open(source).convert('RGB')
    nbScale = 7
    coarseIter = 10000
    coarsetolerance = 0.05
    minSize = 400
    imageNet = True # we can also use MOCO feature here
    scaleR = 1.2 
    coarseModel = CoarseAlign(nbScale, coarseIter, coarsetolerance, 'Homography', minSize, 1, True, imageNet, scaleR)

    coarseModel.setSource(I1)
    coarseModel.setTarget(I2)

    I2w, I2h = coarseModel.It.size
    featt = F.normalize(network['netFeatCoarse'](coarseModel.ItTensor))
                
    #### -- grid     
    gridY = torch.linspace(-1, 1, steps = I2h).view(1, -1, 1, 1).expand(1, I2h,  I2w, 1)
    gridX = torch.linspace(-1, 1, steps = I2w).view(1, 1, -1, 1).expand(1, I2h,  I2w, 1)
    grid = torch.cat((gridX, gridY), dim=3)
    warper = tgm.transform.HomographyWarper(I2h,  I2w)

    bestPara, InlierMask = coarseModel.getCoarse(np.zeros((I2h, I2w)))
    bestPara = torch.from_numpy(bestPara).unsqueeze(0)

    flowCoarse = tgm.transform.warp_grid(warper.grid,bestPara)
    I1_coarse = F.grid_sample(coarseModel.IsTensor, flowCoarse)
    I1_coarse_pil = transforms.ToPILImage()(I1_coarse.cpu().squeeze())

    featsSample = F.normalize(network['netFeatCoarse'](I1_coarse))


    corr12 = network['netCorr'](featt, featsSample)
    flowDown8 = network['netFlowCoarse'](corr12, False) ## output is with dimension B, 2, W, H

    flowUp = F.interpolate(flowDown8, size=(grid.size()[1], grid.size()[2]), mode='bilinear')
    flowUp = flowUp.permute(0, 2, 3, 1)

    flowUp = flowUp + grid

    flow12 = F.grid_sample(flowCoarse.permute(0, 3, 1, 2), flowUp).permute(0, 2, 3, 1).contiguous()

    I1_fine = F.grid_sample(coarseModel.IsTensor, flow12)
    I1_fine_pil = transforms.ToPILImage()(I1_fine.cpu().squeeze())

    final_image=get_Avg_Image(I1_fine_pil, coarseModel.It)
    return final_image


if __name__=="__main__":
    pass



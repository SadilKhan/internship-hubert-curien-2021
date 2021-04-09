import numpy as np
import pandas as pd
import cv2
from json_to_dataset import JsonToCSV
from datagenerator import DataGenerator
from iou import bb_intersection_over_union

class TemplateMatcher:

    """ Template Matching """

    def __init__(self,json_file,slack=0):

        """
        Params:

        json_file --> JSON file created using labelme.
        label_name --> INT or STRING. Label_name for which Template Matching will be used
        slack --> INT. Variable to adjust the bounding box

        """

        self.json=json_file
        self.slack=slack

        # Transform the JSON to CSV
        json2csv=JsonToCSV(self.json_file)

        # Saving the original Image File
        self.image=np.asarray(json2csv.image,dtype=np.uint8)

        # Obtaining The dataset
        self.data=json2csv.dataset        

    
    def find_all_template(self,image,template,method=cv2.TM_CCOEFF_NORMED,threshold):
        """ Find all the the boxes for the template in image """

       self.w,self.h=template.shape[::-1]
       self.res=cv2.matchTemplate(image,template,method=method)
       self.loc = np.where( res >= threshold)

        # Apply Non Max Suppression to delete Overlapping boxes
        self.boxes=[]
        for pt in zip(*self.loc[::-1]):
            self.boxes.append(pt)

       self.boxes=self.non_max_suppression(sorted(self.boxes))
       return self.boxes
    
    def create_boxes(self,box):
        return [[box[0],box[1]],[box[0]+self.w,box[1]+self.h]]

    def non_max_suppression(self,boxes):
        new_boxes=[boxes[0]]
        box=self.create_boxes(new_boxes[-1])
        
        for b in boxes:
            present_box=self.create_boxes(b)
            if bb_intersection_over_union(box,present_box)<0.001:
                new_boxes.append(b)
                box=self.create_boxes(b)
        return new_boxes
    
    def plot_image(self,figsize=(20,20)):
        for pt in zip(*loc[::-1]):
            cv2.rectangle(self.image, pt, (pt[0] + h, pt[1] + w), (0,255,255), 2)
        plt.figure(figsize=figsize)
        plt.imshow(self.image,cmap="gray")

        

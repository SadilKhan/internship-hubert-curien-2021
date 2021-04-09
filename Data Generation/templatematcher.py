import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sys
import json
from json_to_dataset import JsonToCSV
from datagenerator import DataGenerator
from iou import bb_intersection_over_union

class TemplateMatcher(DataGenerator):

    """ Template Matching """

    def __init__(self,json_file,slack_w=[0,0],slack_h=[0,0]):
        super().__init__()

        """
        Params:

        json_file --> JSON file created using labelme.
        slack_w --> INT. Variable to adjust the width of bounding box.
        slack_h --> INT. Variable to adjust the height of bounding box.
        """

        self.json_file=json_file
        self.slack_w_left=slack_w[0]
        self.slack_w_right=slack_w[1]
        self.slack_h_up=slack_h[0]
        self.slack_h_bottom=slack_h[1]

        # Transform the JSON to CSV
        json2csv=JsonToCSV(self.json_file)

        # Saving the original Image File
        self.image=json2csv.image

        # The Image file needs to be transformed to Grayscale.
        self.image=np.asarray(self.rgb2gray(self.image),dtype=np.uint8)

        # Obtaining The dataset
        self.data=json2csv.dataset


        # To store the height and width of every template boxes

        self.height=dict()
        self.width=dict()

    def match_template(self,label,threshold):
        """ Template Matcher for Any Label """
        # Creating a dictionary for storing boxes for all images
        self.all_boxes=dict() 
        if label=="all":
            # Get All unique labels
            self.all_labels=list(self.data['label'].unique())
        else:
            self.all_labels=list(label)

        for l in self.all_labels:
            # bounding box for the object
            bbox=self.data[self.data['label']==l]['bbox'].iloc[0]
            # The image template that we need to match
            template=self.image[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            w,h=template.shape[::-1]
            self.height[l]=h
            self.width[l]=w


            # All the detected objects for the template
            self.all_boxes[l]=self.find_all_template(template,threshold)
        return self.all_boxes
    
    def find_all_template(self,template,threshold=0.25,method=cv2.TM_CCOEFF_NORMED):
        """ Find all the the boxes for the template in image """
        self.w,self.h=template.shape[::-1]
        self.res=cv2.matchTemplate(self.image,template,method=method)
        self.loc = np.where(self.res >= threshold)

        # Apply Non Max Suppression to delete Overlapping boxes
        self.boxes=[]
        for pt in zip(*self.loc[::-1]):
            pt=(pt[0]-self.slack_w,pt[1]+self.slack_h)
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
        for k in self.all_labels:
            box_k=self.all_boxes[k]
            h=self.height[k]
            w=self.width[k]

            for pt in box_k:
                cv2.rectangle(self.image, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        plt.figure(figsize=figsize)
        plt.imshow(self.image,cmap="gray")
        plt.show()

    def save_json(self,name,json_file):

        for key in list(json_file.keys()):
            json_file[key]=str(json_file[key])
        with open(name,"w+") as f:
            json.dump(json_file,f)

    def save_csv(self,dir_name,dict_data):

        """ Save the file in the CSV """
        pass







if __name__=="__main__":

    """if sys.arg"""

    tm=TemplateMatcher("/Users/ryzenx/Documents/Internship/Dataset/image2.json",slack_w=-10)


    boxes=tm.match_template("all",0.25)
    #tm.save_json("automate.json",boxes)
    tm.plot_image((10,10))    
        

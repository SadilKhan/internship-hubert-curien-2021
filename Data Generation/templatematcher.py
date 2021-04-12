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
        slack_w --> LIST. Variable to adjust the width of bounding box.
        slack_h --> LIST. Variable to adjust the height of bounding box.
        """

        self.json_file=json_file
        self.slack_w=slack_w
        self.slack_h=slack_h
        self.slack_w_left=self.slack_w[0]
        self.slack_w_right=self.slack_w[1]
        self.slack_h_up=self.slack_h[0]
        self.slack_h_bottom=self.slack_h[1]

        # Transforming the JSON to CSV
        json2csv=JsonToCSV(self.json_file)

        # Store the original labelme json file
        self.labelmeData=json2csv.json_data

        # Saving the original Image File
        self.image=json2csv.image

        # The Image file needs to be transformed to Grayscale.
        self.image=np.asarray(self.rgb2gray(self.image),dtype=np.uint8)

        # Obtaining The dataset
        self.data=json2csv.dataset


        # To store the height and width of every template boxes

        self.height=dict()
        self.width=dict()


        # Print the Arguments
        print(f"TemplateMatcher(json_file={self.json_file},slack_w={self.slack_w},slack_h={self.slack_h}")

    def match_template(self,label,threshold):
        """ Template Matcher for Any Label """
        # Creating a dictionary for storing boxes for all images
        self.all_boxes=dict() 
        if label=="-":
            label="all"
        if threshold=="-":
            threshold=0.25

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
            self.height[l]=h+self.slack_h_bottom
            self.width[l]=w+self.slack_w_right


            # All the detected objects for the template
            self.all_boxes[l]=self.find_all_template(template,threshold)
        return self.all_boxes
    
    def find_all_template(self,template,threshold=0.25,method=cv2.TM_CCOEFF_NORMED):
        """ Find all the the boxes for the template in image """

        self.w,self.h=template.shape[::-1]
        self.w=self.w+self.slack_w_right
        self.h=self.h+self.slack_h_bottom

        self.res=cv2.matchTemplate(self.image,template,method=method)
        self.loc = np.where(self.res >= threshold)

        # Apply Non Max Suppression to delete Overlapping boxes
        self.boxes=[]
        for pt in zip(*self.loc[::-1]):
            pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
            self.boxes.append(pt)

        self.boxes=self.non_max_suppression(sorted(self.boxes))
        self.boxes=[self.create_boxes(i) for i in self.boxes]
        return self.boxes
    
    def create_boxes(self,box):
        """ Create four coordinates from the corner point """

        return [[box[0],box[1]],[box[0]+self.w,box[1]+self.h]]

    def non_max_suppression(self,boxes):

        """ Perform Non-Max Suppression for optimal bounding boxes"""

        new_boxes=[boxes[0]]
        box=self.create_boxes(new_boxes[-1])
        
        for b in boxes:
            present_box=self.create_boxes(b)
            if bb_intersection_over_union(box,present_box)<0.001:
                new_boxes.append(b)
                box=self.create_boxes(b)
        return new_boxes
    
    def plot_image(self,figsize=(20,20)):

        """ Plot the Image with the bounding boxes """

        for k in self.all_labels:
            box_k=self.all_boxes[k]
            h=self.height[k]
            w=self.width[k]

            for pt in box_k:
                cv2.rectangle(self.image, (pt[0][0],pt[0][1]), (pt[1][0],pt[1][1]), (0,255,255), 2)
        plt.figure(figsize=figsize)
        plt.imshow(self.image,cmap="gray")
        plt.show()

    def createJSON(self):
        """ A class to transform JSON or CSV file to LabelMe JSON format """
        # Store all the keys.
        keys=self.labelmeData.keys()

        # Store all the labels from all_boxes
        labels=self.all_boxes.keys() 

        # Append all the all boxes.
        for lb in labels:
            for bx in self.all_boxes[lb]:
                # A temporary Dictionary
                temp=dict()
                temp['label']=lb
                temp['line_color']=None
                temp['fill_color']=None
                bx=[[int(bx[0][0]),int(bx[0][1])],[int(bx[1][0]),int(bx[1][1])]]
                temp['points']=bx
                temp['shape_type']="rectangle"
                self.labelmeData['shapes'].append(temp)
        
        # Store the json file.
        with open(self.json_file, 'w+') as fp:
            json.dump(self.labelmeData, fp,indent=2)

    def save_csv(self,dir_name,dict_data):

        """ Save the file in the CSV format """

        pass



if __name__=="__main__":

    if len(sys.argv)<2:
        print("\nJson_Directory : STRING, The Directory file for Json File,\n")
        print("Threshold: FLOAT, The threshold value for matchTemplate. Lower the value, more bounding boxes. Default 0.25\n")
        print("Label: STRING or INT, The label for which the bounding boxes are to calculated. Default all.\n")
        print("Plot Image: BOOLEAN, plot the image with the bounding boxes. Default True\n")
        print("Save: JSON/FALSE, To save the bounding boxes in JSON. Default False\n")
        print("slack_w(Optional): LIST, Parameter to adjust the width of the bounding box. Type - for default value\n")
        print("slack_h(Optional): LIST, Parameter to adjust the height of the bounding box. Type - for default value\n ")

    elif len(sys.argv)==2:
        json_file=sys.argv[1]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template("all",0.25)

    elif len(sys.argv)==3:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template("all",threshold)

    elif len(sys.argv)==4:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold)

    elif len(sys.argv)==5:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold)
        tm.plot_image(figsize=(20,20))   

    elif len(sys.argv)==6:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        plot=sys.argv[4]
        save=sys.argv[5]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold=threshold)
        if plot:
            tm.plot_image(figsize=(20,20))
        
        if save:
            tm.createJSON()

    elif len(sys.argv)==7:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        plot=sys.argv[4]
        save=sys.argv[5]
        slack_w=sys.argv[6]
        tm=TemplateMatcher(json_file,slack_w=slack_w)
        boxes=tm.match_template(label,threshold=threshold)
        if plot:
            tm.plot_image(figsize=(20,20))
        
        if save:
            tm.createJSON()
    elif len(sys.argv)==7:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        plot=sys.argv[4]
        save=sys.argv[5]
        slack_w=sys.argv[6]
        if slack_w=="-":
            slack_w=[0,0]
        slack_h=sys.argv[7]
        tm=TemplateMatcher(json_file,slack_w=slack_w,slack_h=slack_h)
        boxes=tm.match_template(label,threshold=threshold)
        if plot:
            tm.plot_image(figsize=(20,20))
        
        if save:
            tm.createJSON() 

        

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import sys
import json
import warnings
from error_message import DuplicateWarning
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

        # Image Name
        self.imagePath=self.labelmeData.get("imagePath")


        # To store the height and width of every template boxes

        self.height=dict()
        self.width=dict()


        # Print the Arguments
        print(f"TemplateMatcher(json_file={self.json_file},slack_w={self.slack_w},slack_h={self.slack_h}")

    def match_template(self,label,threshold,search_space_boundary=0,rotation_min=None,rotation_max=None,flipping=False):
        """ Template Matcher for Any Label """
        print("TEMPLATE MATCHING STARTED....")
        # Creating a dictionary for storing boxes for all images
        self.all_boxes=dict() 

        # A dictionary to store the cross-correlation value
        self.all_box_dict=dict()
        if label=="-":
            label="all"
        if threshold=="-":
            threshold=0.25
        else:
            threshold=float(threshold)

        if label=="all":
            # Get All unique labels
            self.all_labels=list(self.data['label'].unique())

            # Raise a warning if there are duplicate labels
            if len(self.all_labels)!=len(self.data['label']):
                self.warning()
            
            # Check if the label is integer
            for lb in self.all_labels:
                self.check_int(lb)
        else:
            self.all_labels=list(label)
        
        # If range of rotation is provided or only the rotation
        if not rotation_min and not rotation_max:
            rotation_range=(None,None)
        elif rotation_min and not rotation_max:
            rotation_range=(rotation_min,None)
        else:
            rotation_range=(rotation_min,rotation_max)


        for l in self.all_labels:
            bbox=self.data[self.data['label']==l]['bbox'].iloc[0]
            # The image template that we need to match
            template=self.image[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]
            w,h=template.shape[::-1]
            self.height[l]=h+self.slack_h_bottom
            self.width[l]=w+self.slack_w_right


            # All the detected objects for the template
            self.all_boxes[l],self.all_box_dict[l]=self.find_all_template(template,l,threshold,search_space_boundary,rotation_range,flipping)
        print("TEMPLATE MATCHING ENDED")

        return self.all_boxes
    
    def find_all_template(self,template,label="0",threshold=0.25,s=0,rotation_range=(None,None),flipping=False):
        """ Find all the the boxes for the template in image """

        self.w,self.h=template.shape[::-1]
        self.w=self.w+self.slack_w_right
        self.h=self.h+self.slack_h_bottom
        self.threshold=threshold

        # The provided box for the label
        bbox=self.data[self.data['label']==label]["bbox"].iloc[0]

        # Create the rotation angle spaces
        if_rotation=True
        if rotation_range==(None,None):
            if_rotation=False
        if not rotation_range[-1]:
            rotaion_space= [rotation_range[0]]
        else:
            rotaion_space=list(range(rotation_range[0],rotation_range[1],1))

        # Restrict the search space
        if s<=10:
            search_space=self.image[bbox[0][1]-s*self.h:bbox[1][1]+s*self.h,:]

            # The starting coordintaes of the height.
            self.h_start=bbox[0][1]-s*self.h
        if s>10:
            search_space=self.image[bbox[0][1]-s:bbox[1][1]+s,:]

            # The starting coordintaes of the height.
            self.h_start=bbox[0][1]-s

        # INITIALIZING VARIABLE TO CREATE AND SELECT BOXES
        self.boxes=[]
        self.box_dict=dict()

        # Match Translation
        a,b=self.match(search_space,label,template)
        # Match Rotation
        if if_rotation:
            for j in rotaion_space:
                template_rotated=np.array(self.rotate_image(template,j))
                a,b=self.match(search_space,label,template_rotated)
        
        # Match Flipping:
        if flipping:
            label_flipped=label+"_flipped"
            template_flipped=np.array(self.flip_image(template))
            a,b=self.match(search_space,label_flipped,template_flipped)

            template_mirrored=np.array(self.mirror_image(template))
            a,b=self.match(search_space,label_flipped,template_mirrored)

        # Match Scaled template of the flipped template
        for scale in np.linspace(0.8,2,25):
            w,h=template.shape[::-1]
            w=int(np.floor(w*scale))
            # Scaling the Template
            template_scaled=np.array(self.resize_image(template,(w,h)))
            a,b=self.match(search_space,label,template_scaled)

            if flipping:
                template_flipped_scaled=np.array(self.resize_image(template_flipped,(w,h)))
                a,b=self.match(search_space,label_flipped,template_flipped_scaled)
                template_mirrored_scaled=np.array(self.resize_image(template_mirrored,(w,h)))
                a,b=self.match(search_space,label_flipped,template_mirrored_scaled)
        
        self.non_max_suppression()
        self.boxes=[self.create_boxes(i) for i in self.boxes]
        print(f"TEMPLATE MATCHING FINISHED FOR LABEL {label}")
        return self.boxes,self.box_dict
    
    def match(self,search_space,label,template,method=cv2.TM_CCOEFF_NORMED):
        """ Template Matching for Translation, Rotation, Flipping """

        res=cv2.matchTemplate(search_space,template,method=method)
        loc = np.where(res >= self.threshold)
        loc=(loc[0]+self.h_start,loc[1])

        w,h=template.shape[::-1]
        # Collect all the top left corners of the boxes
        # Boxes only for translation matching
        boxes=[]
        for pt in zip(*loc[::-1]):
            pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
            boxes.append(pt)
            self.box_dict[pt]=[res[pt[1]-self.h_start][pt[0]],(w,h),label]

        self.boxes+=boxes

        return res,boxes
    
    def create_boxes(self,box):
        """ Create four coordinates from the corner point """
        w,h=self.box_dict[(box[0],box[1])][1]
        return [[box[0],box[1]],[box[0]+w,box[1]+h]]
    
    def warning(self):
        warnings.warn("WARNING: Duplicate Label Detected. First Box of the same label will be taken into account.", DuplicateWarning)
    
    def check_int(self,value):
        if not type(value) is int:
            raise TypeError("Only Integer Values are allowed")

    def non_max_suppression(self):
        
        """ Perform Non-Max Suppression for removing overlapping boxes."""

        new_box=[self.boxes[0]]
        for i,b in enumerate(self.boxes):
            b_box=self.create_boxes(b)
            update=True
            for j,nb in enumerate(new_box):
                nb_box=self.create_boxes(nb)
                if bb_intersection_over_union(nb_box,b_box) > 0.5 and self.box_dict[b][0]>self.box_dict[nb][0]:
                    new_box[j]=b
                    update=False
                    break
                if bb_intersection_over_union(nb_box,b_box) > 0.1:
                    update=False
                    break

            if update:
                new_box.append(b)
        self.boxes=new_box
        return new_box
    
    def plot(self,figsize=(20,20)):


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

    def random_color(self):
        """ Generate Random colors for bounding box"""
        levels = random.choice([0,1,2])
        color=[]
        for i in range(3):
            if i==levels:
                color.append(random.choice(range(120,255,20)))
            else:
                color.append(random.choice(range(5,60,2)))
        
        return color+[128]
        
    def createJSON(self):
        """ A class to transform JSON or CSV file to LabelMe JSON format """
        # Store all the keys.
        keys=self.labelmeData.keys()

        # Store all the labels from all_boxes
        labels=list(self.all_boxes.keys())

        # Get all the colors
        shapes=self.labelmeData['shapes']
        
        colors=dict()
        for i in range(len(labels)):
            color=shapes[i]['line_color']
            if not color:
                color=self.random_color()
                #color=[int(i) for i in color]
                colors[labels[i]]=color
            else: 
                try:
                    colors[labels[i]]=color
                except:
                    pass
        
        self.labelmeData['shapes']=[]

        # Append all the all boxes.
        for lb in labels:
            for bx in self.all_boxes[lb]:
                # A temporary Dictionary
                temp=dict()
                try:
                    temp['label']=self.imagePath.split(" / ")[-1].split(".")[0]+"_"+self.all_box_dict[lb][tuple(bx[0])][2]
                except:
                    temp['label']=lb
                temp['line_color']=colors[lb]
                temp['fill_color']=None
                bx=[[int(bx[0][0]),int(bx[0][1])],[int(bx[1][0]),int(bx[1][1])]]
                temp['points']=bx
                temp['shape_type']="rectangle"
                self.labelmeData['shapes'].append(temp)
        
        # Store the json file.
        with open(self.json_file, 'w+') as fp:
            json.dump(self.labelmeData, fp,indent=2)
        print("SAVED THE JSON FILE. OPEN LabelMe and REOPEN THE SAME JSON")

    def save_csv(self,dir_name,dict_data):

        """ Save the file in the CSV format """

        pass



if __name__=="__main__":

    if len(sys.argv)<2:
        print("\n1. Json_Directory : STRING, The Directory file for Json File.\n")
        print("2. Threshold: FLOAT, The threshold value for matchTemplate. Lower the value, more bounding boxes. Default 0.25.\n")
        print("3. Label: STRING or INT, The label for which the bounding boxes are to calculated. Default all.\n")
        print("4. Search Space Boundary: FLOAT. Default 0. If s is the value, range of the space [[a,b],[c,d]] is from (b-s*(d-b),d+s*(d-b)) If s>10,then it's simply added with b and d.\n")
        print("5. Plot Image: BOOLEAN, plot the image with the bounding boxes. Default True.\n")
        print("6. Save: Boolean, To save the bounding boxes in JSON. Write - for default(False).\n")
        print("7. Rotation: LIST/False. To check if the rotated template matches anything in Image. If list then a range of angle must be given. For specific angle, just give one value and None for the other one.\n")
        print("8. Flipping: True/False. To check if the rotated template matches anything in Image.\n")
        print("9. slack_w(Optional): LIST, Parameter to adjust the width of the bounding box. Type - for default value.\n")
        print("10. slack_h(Optional): LIST, Parameter to adjust the height of the bounding box. Type - for default value.\n ")

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
        search_space_boundary=int(sys.argv[4])
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold,search_space_boundary=search_space_boundary)

    elif len(sys.argv)==6:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])
        if search_space_boundary=="-":
            search_space_boundary=0
        
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold,search_space_boundary=search_space_boundary)
        if sys.argv[5]=="True":
            tm.plot(figsize=(20,20))   

    elif len(sys.argv)==7:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])
        if search_space_boundary=="-":
            search_space_boundary=0
        plot=sys.argv[5]
        save=sys.argv[6]
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold=threshold,search_space_boundary=search_space_boundary)
        if plot=="True":
            tm.plot(figsize=(20,20))
        
        if save=="True":
            tm.createJSON()
    
    elif len(sys.argv)==8:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])
        if search_space_boundary=="-":
            search_space_boundary=0
        plot=sys.argv[5]
        save=sys.argv[6]
        rotation=sys.argv[7]
        if rotation!="-":
            rotation=list(map(str, rotation.strip('[]').split(',')))
            try:
                rotation_min=int(rotation[0])
                rotation_max=int(rotation[1])
            except:
                rotation_min=int(rotation[0])
                rotation_max=None
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold=threshold,search_space_boundary=search_space_boundary,rotation_min=rotation_min,rotation_max=rotation_max)

        if plot=="True":
            tm.plot(figsize=(20,20))
        
        if save=="True":
            tm.createJSON()
    
    elif len(sys.argv)==9:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])

        if search_space_boundary=="-":
            search_space_boundary=0

        plot=sys.argv[5]
        save=sys.argv[6]
        rotation=sys.argv[7]
        flipping=sys.argv[8]
        if flipping=="True":
            flipping=True
        else:
            flipping=False

        if rotation!="-":
            rotation=list(map(str, rotation.strip('[]').split(',')))
            try:
                rotation_min=int(rotation[0])
                rotation_max=int(rotation[1])
            except:
                rotation_min=int(rotation[0])
                rotation_max=None
        else:
            rotation_max=None
            rotation_min=None
    
        tm=TemplateMatcher(json_file)
        boxes=tm.match_template(label,threshold=threshold,search_space_boundary=search_space_boundary,rotation_min=rotation_min,rotation_max=rotation_max,flipping=flipping)

        if plot=="True":
            tm.plot(figsize=(20,20))
        
        if save=="True":
            tm.createJSON()


    elif len(sys.argv)==10:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])

        if search_space_boundary=="-":
            search_space_boundary=0

        plot=sys.argv[5]
        save=sys.argv[6]
        rotation=sys.argv[7]
        flipping=sys.argv[8]
        if flipping=="True":
            flipping=True
        else:
            flipping=False

        if rotation!="-":
            rotation=list(map(str, rotation.strip('[]').split(',')))
            try:
                rotation_min=int(rotation[0])
                rotation_max=int(rotation[1])
            except:
                rotation_min=int(rotation[0])
                rotation_max=None
        else:
            rotation_max=None
            rotation_min=None
        
        slack_w=sys.argv[9]
        slack_w = list(map(float, slack_w.strip('[]').split(',')))

        tm=TemplateMatcher(json_file,slack_w=slack_w)
        boxes=tm.match_template(label,threshold=threshold,search_space_boundary=search_space_boundary,rotation_min=rotation_min,rotation_max=rotation_max,flipping=flipping)

        if plot=="True":
            tm.plot(figsize=(20,20))
        
        if save=="True":
            tm.createJSON()

    elif len(sys.argv)==11:
        json_file=sys.argv[1]
        threshold=sys.argv[2]
        label=sys.argv[3]
        search_space_boundary=int(sys.argv[4])

        if search_space_boundary=="-":
            search_space_boundary=0

        plot=sys.argv[5]
        save=sys.argv[6]
        rotation=sys.argv[7]
        flipping=sys.argv[8]
        if flipping=="True":
            flipping=True
        else:
            flipping=False

        if rotation!="-":
            rotation=list(map(str, rotation.strip('[]').split(',')))
            try:
                rotation_min=int(rotation[0])
                rotation_max=int(rotation[1])
            except:
                rotation_min=int(rotation[0])
                rotation_max=None
        else:
            rotation_max=None
            rotation_min=None
        
        slack_w=sys.argv[9]
        if slack_w=="-":
            slack_w=[0,0]
        else:
            slack_w = list(map(float, slack_w.strip('[]').split(',')))

        slack_h=sys.argv[10]
        slack_h = list(map(float, slack_h.strip('[]').split(',')))
        
        tm=TemplateMatcher(json_file,slack_w=slack_w,slack_h=slack_h)
        boxes=tm.match_template(label,threshold=threshold,search_space_boundary=search_space_boundary,rotation_min=rotation_min,rotation_max=rotation_max,flipping=flipping)

        if plot=="True":
            tm.plot(figsize=(20,20))
        
        if save=="True":
            tm.createJSON()

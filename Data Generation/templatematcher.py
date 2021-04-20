import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
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

    def match_template(self,label,threshold,search_space_boundary=0,rotation_min=None,rotation_max=None,flipping=False):
        """ Template Matcher for Any Label """
        print("TEMPLATE MATCHING STARTED....")
        # Creating a dictionary for storing boxes for all images
        self.all_boxes=dict() 
        if label=="-":
            label="all"
        if threshold=="-":
            threshold=0.25
        else:
            threshold=float(threshold)

        if label=="all":
            # Get All unique labels
            self.all_labels=list(self.data['label'].unique())
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
            self.all_boxes[l]=self.find_all_template(template,l,threshold,search_space_boundary,rotation_range,flipping)
        print("TEMPLATE MATCHING ENDED")

        return self.all_boxes
    
    def find_all_template(self,template,label=None,threshold=0.25,s=0,rotation_range=(None,None),flipping=False,method=cv2.TM_CCOEFF_NORMED):
        """ Find all the the boxes for the template in image """

        self.w,self.h=template.shape[::-1]
        self.w=self.w+self.slack_w_right
        self.h=self.h+self.slack_h_bottom

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
            h_start=bbox[0][1]-s*self.h
        if s>10:
            search_space=self.image[bbox[0][1]-s:bbox[1][1]+s,:]

            # The starting coordintaes of the height.
            h_start=bbox[0][1]-s


        # For Translation
        self.res_tr=cv2.matchTemplate(search_space,template,method=method)
        self.loc = np.where(self.res_tr >= threshold)
        self.loc=(self.loc[0]+h_start,self.loc[1])

        # Apply Non Max Suppression to delete Overlapping boxes
        self.boxes=[]
        for pt in zip(*self.loc[::-1]):
            pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
            self.boxes.append(pt)

        # Boxes only for translation
        #self.boxes_tr=self.non_max_suppression(sorted(self.boxes),True)
        self.boxes_tr=self.delete_overlapping_boxes(self.boxes,self.boxes)
        self.boxes=self.boxes_tr
        


        
        # Do several rotations of the template and find matches
        if if_rotation:
            self.boxes_r=[]
            for j in rotaion_space:
                print(f"Rotation {j}\n")
                template=np.array(self.rotate_image(template,j))
                self.res_r=cv2.matchTemplate(self.image,template,method=method)
                self.loc = np.where(self.res_r >= 0.25)
                for pt in zip(*self.loc[::-1]):
                    pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
                    self.boxes_r.append(pt)
                
                if len(self.boxes_r)>0:
                    # Non - Max supression 
                    self.boxes_r=self.delete_overlapping_boxes(sorted(self.boxes_r))
                    # Delete all those boxes overlapping with boxes found in translation.
                    self.boxes=self.boxes+self.delete_overlapping_boxes(self.boxes,self.boxes_r)



        # Flip Image horizontally and vertically
        if flipping:
            self.boxes_f=[]
            # Flip Image
            template=np.array(self.flip_image(template))
            
            self.res_f=cv2.matchTemplate(search_space,template,method=method)
            self.loc = np.where(self.res_f >= 0.15)
            self.loc=(self.loc[0]+h_start,self.loc[1])

            for pt in zip(*self.loc[::-1]):
                pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
                self.boxes_f.append(pt)
            if len(self.boxes_f)>0:
                # Non Max Suppression
                self.boxes_f=self.delete_overlapping_boxes(sorted(self.boxes_f))
                # Delete all those boxes overlapping with boxes found in translation.
                self.boxes=self.boxes+self.delete_overlapping_boxes(self.boxes,self.boxes_f)


            # Vertical Image
            self.boxes_m=[]
            template=np.array(self.mirror_image(template))
            self.res_m=cv2.matchTemplate(search_space,template,method=method)
            self.loc = np.where(self.res_m >= 0.15)

            self.loc=(self.loc[0]+h_start,self.loc[1])
            for pt in zip(*self.loc[::-1]):
                pt=(pt[0]+self.slack_w_left,pt[1]+self.slack_h_up)
                self.boxes_m.append(pt)

            if len(self.boxes_m)>0:
                # Non Max Suppression
                self.boxes_m=self.delete_overlapping_boxes(sorted(self.boxes_m))
                # Delete all those boxes overlapping with boxes found in translation.
                self.boxes=self.boxes+self.delete_overlapping_boxes(self.boxes,self.boxes_m)
        
        # Create Boxes
        self.boxes=self.delete_overlapping_boxes(self.boxes)
        self.boxes=[self.create_boxes(i) for i in self.boxes]
        print(f"TEMPLATE MATCHING FINISHED FOR LABEL {label}")
        return self.boxes
    
    def create_boxes(self,box):
        """ Create four coordinates from the corner point """

        return [[box[0],box[1]],[box[0]+self.w,box[1]+self.h]]

    def non_max_suppression(self,boxes,for_translation=False):

        """ Perform Non-Max Suppression for removing overlapping boxes. For Translation boxes, it will a bit different."""

        new_boxes=[boxes[0]]
        box=self.create_boxes(new_boxes[-1])
        
        for b in boxes:
            update=True
            present_box=self.create_boxes(b)
            for bx in new_boxes:
                box=self.create_boxes(bx)
                """if for_translation:
                if bb_intersection_over_union(box,present_box)>0.01:
                    update=False
                    break         
                else:"""
                if bb_intersection_over_union(box,present_box)>0.00000001:
                    update=False
                    break
            if update:
                new_boxes.append(b)
                box=self.create_boxes(b)
        return new_boxes
    
    def delete_overlapping_boxes(self,A,B=None,res_A=self.boxes_tr,res_B=None):
        """Specific Non-max Suppression. Delete boxes of B which overlaps with A. Returns transformed B """

        new_B=[]

        if not B:
            B=A
        
        if B==A:
            for i in range(len(B)):
                b=B[i]
                b_box=self.create_boxes(b)
                update=True
                for j in range(i+1,len(A)):
                    a=A[j]
                    a_box=self.create_boxes(a)
                    if bb_intersection_over_union(a_box,b_box)>0.01:
                        update=False
                        break

                if update:
                    new_B.append(b)
            return new_B

        else:
            for b in B:
                b_box=self.create_boxes(b)
                update=True
                for a in A:
                    a_box=self.create_boxes(a)
                    if res_A and res_B:
                        if bb_intersection_over_union(a_box,b_box)>0.01 and res_A[a[1],a[0]]>=res_B[b[1],b[0]]:
                            update=False
                            break
                    else:
                        if bb_intersection_over_union(a_box,b_box)>0.01:
                            update=False
                            break

                if update:
                    new_B.append(b)            
            return new_B

    
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

    def createJSON(self):
        """ A class to transform JSON or CSV file to LabelMe JSON format """
        # Store all the keys.
        keys=self.labelmeData.keys()

        # Store all the labels from all_boxes
        labels=list(self.all_boxes.keys())

        # Get all the colors
        shapes=self.labelmeData['shapes']
        
        colors=dict()
        for i in range(len(shapes)):
            color=shapes[i]['line_color']
            if not color:
                color=list(np.random.randint(0,255,3))+[128]
                color=[int(i) for i in color]
                colors[labels[i]]=color
            else:
                colors[labels[i]]=color
        
        self.labelmeData['shapes']=[]

        # Append all the all boxes.
        for lb in labels:
            for bx in self.all_boxes[lb]:
                # A temporary Dictionary
                temp=dict()
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
        print("6. Save: JSON/FALSE, To save the bounding boxes in JSON. Default False.\n")
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

        

if __name__=="__main__":
    """tm=TemplateMatcher("/Users/ryzenx/Documents/Internship/Dataset/image1.json",slack_w=[0,0])
    boxes=tm.match_template("1",0.25,1)
    tm.plot(figsize=(10,10))"""

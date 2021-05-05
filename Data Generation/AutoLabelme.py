from tkinter import *
import numpy as np
from tkinter.constants import DISABLED, NORMAL
import tkinter as tk
from platform import system
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk,Image,ImageDraw
from templatematcher import TemplateMatcher
import os
import shutil
import random
import sys
import json
import warnings
import colorsys
platformD = system()
if platformD == 'Darwin':
    from tkmacosx import Button



CUR_DIR=os.curdir

class GUI():

    def __init__(self):
        # Create a gui
        self.root=Tk()
        self.root.title("Auto Image Annotator")
        self.root.geometry("2500x1600")

        # Menu
        self.menu()

        # Frame

    def menu(self):
        # Main menu bar
        self.my_menu=Menu(self.root)

        # File Menu
        self.file_menu=Menu(self.my_menu)
        self.my_menu.add_cascade(label="File",menu=self.file_menu)
        self.file_menu.add_command(label="Open JSON",command=self.open_json_file)
        self.file_menu.add_command(label="Restart",command=self.restart)
        self.file_menu.add_command(label="Exit",command=self.quit)

        # Edit Menu
        self.edit_menu=Menu(self.my_menu)
        self.my_menu.add_cascade(label="Edit",menu=self.edit_menu)
        self.edit_menu.add_command(label="Undo",state=DISABLED)

        self.root.config(menu=self.my_menu)
    
    def frame(self,image):
        self.image_frame=Frame(self.root,width=1600,height=1000)
        self.my_image=ImageTk.PhotoImage(image)
        self.label=Label(self.image_frame,image=self.my_image)
        self.image_frame.place(x=10,y=10)
        self.label.place(x=11,y=11)
        
        

    def restart(self):
        self.image_frame.place_forget()
        self.less_button.place_forget()
        self.more_button.place_forget()
        self.start_button.place_forget()
        self.next_button.place_forget()
        self.prev_button.place_forget()
        self.refine_button.place_forget()
        self.finer_resize_button.place_forget()
 
    
    def quit(self):
        message=messagebox.askyesno("My Popup","Do you want to exit?")
        if message:
            self.root.quit()
    
    def less_boxes(self):
        label=self.all_labels[self.label_no]
        # Increasing the threshold will result in less boxes
        threshold=self.all_threshold[label]+0.05
        self.all_threshold[label]=threshold
        self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.image_frame.place_forget()
        self.frame(image)
        self.button()
        self.prev_button["state"]=NORMAL
        self.start_button["state"]=DISABLED
        
    
    def more_boxes(self):
        label=self.all_labels[self.label_no]
        # Decreasing the threshold will result in more boxes
        threshold=self.all_threshold[label]-0.05
        self.all_threshold[label]=threshold
        self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.image_frame.place_forget()
        self.frame(image)
        self.button()
        self.prev_button["state"]=NORMAL
        self.start_button["state"]=DISABLED

    def check_boxes_label(self,label):
        """ Check if all the labels are flipped or not"""
        bx=self.all_boxes[label]
        box_dict=self.all_box_dict[label]
        nbx=len(bx)
        n_flipped=0
        for i in bx:
            if "flipped" == box_dict[(i[0][0],i[0][1])][2].split("_")[-1]:
                n_flipped+=1
        if n_flipped==nbx:
            new_label="_".join(box_dict[(i[0][0],i[0][1])][2].split("_")[:-1])
            return True,new_label
        return False,None
   
    def save(self):
        """ A class to transform JSON or CSV file to LabelMe JSON format """

        # Since the file is saved, self.save is True
        self.saved=True

        # Store all the keys.
        keys=self.jsondata.keys()

        # Store all the labels from all_boxes
        labels=list(self.all_boxes.keys())

        # Get all the colors
        shapes=self.jsondata['shapes']

        # Outline for boxes
        colors=dict()
        for i in range(len(labels)):
            """if i<=self.label_no:
                color=shapes[i]['line_color']"""
            color=self.tm.random_color(int(labels[i]))
            #color=[int(i) for i in color]
            colors[labels[i]]=color
            """if not color:
                color=self.tm.random_color(int(labels[i]))
                #color=[int(i) for i in color]
                colors[labels[i]]=color
            else: 
                try:
                    colors[labels[i]]=color
                except:
                    pass"""
        
        self.jsondata['shapes']=[]

        # Append all the all boxes.
        for lb in labels:
            try:
                # if all the boxes are of flipped category, change it to normal
                all_flipped,new_label=self.check_boxes_label(lb)
                for bx in self.all_boxes[lb]:
                    # A temporary Dictionary
                    temp=dict()
                    if not all_flipped:
                        temp['label']=self.imagePath.split("/")[-1].split(".")[0]+"_"+self.all_box_dict[lb][tuple(bx[0])][2]
                    else:
                        temp["label"]=self.imagePath.split("/")[-1].split(".")[0]+"_"+new_label
                    temp['line_color']=colors[lb]
                    temp['fill_color']=None
                    bx=[[int(bx[0][0]),int(bx[0][1])],[int(bx[1][0]),int(bx[1][1])]]
                    temp['points']=bx
                    temp['shape_type']="rectangle"
                    self.jsondata['shapes'].append(temp)
            except:
                pass
        
        # Store the json file.
        self.json_file_name=self.root.filename.split(".")[0]+"_matched.json"
        with open(self.json_file_name, 'w+') as fp:
            json.dump(self.jsondata, fp,indent=2)
        print("JSON FILE SAVED. REOPEN THE JSON FILE IN LABELME. ",self.json_file_name)
    
    def matching_window(self):
        label=self.all_labels[self.label_no]
        threshold=self.all_threshold[label]
        self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.frame(image)
        self.button()
        self.start_button['state']=DISABLED


    def next_matching(self):
        self.label_no+=1
        label=self.all_labels[self.label_no]
        if len(self.all_boxes[label])==0:   
            threshold=self.all_threshold[label]
            self.all_threshold[label]=threshold
            self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.image_frame.place_forget()
        self.frame(image)
        self.button()
        self.prev_button["state"]=NORMAL
        self.start_button["state"]=DISABLED
        # Disable the next button if there are no more labels
        if self.label_no+1==len(self.all_labels):
            self.next_button["state"]=DISABLED 

    def prev_matching(self):
        self.label_no-=1
        label=self.all_labels[self.label_no]
        if len(self.all_boxes[label])==0:   
            threshold=self.all_threshold[label]
            self.all_threshold[label]=threshold
            self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.image_frame.place_forget()
        self.frame(image)
        self.button()
        
        # Disable the next button if there are no more labels
        if self.label_no!=0:
            self.prev_button["state"]=NORMAL

        self.start_button["state"]=DISABLED

    def finer_less_boxes(self):
        label=self.all_labels[self.label_no]
        # Increasing the threshold will result in less boxes
        threshold=self.all_threshold[label]+0.005
        self.all_threshold[label]=threshold
        self.all_boxes[label],self.all_box_dict[label]=self.tm.match_template(label,threshold,100,None,None,True)
        # Plot the image with the bounding boxes
        image=Image.fromarray(self.tm.original_image.copy())
        w,h=image.width,image.height
        image=image.resize((1000,1000),Image.ANTIALIAS)
        img=ImageDraw.Draw(image)
        for bx in self.all_boxes[label]:
            bx=np.array(bx).reshape(-1)
            bx=bx*np.array([1000])/np.array([w,h,w,h])
            bx=tuple(bx)
            img.rectangle(bx,outline="red")
        self.image_frame.place_forget()
        self.frame(image)
        self.button()
        self.prev_button["state"]=NORMAL
        self.start_button["state"]=DISABLED


    def button(self):
        self.start_button=Button(self.root,text="Start Matching",fg="black",command=self.matching_window,disabledforeground="black")
        self.next_button=Button(self.root,text="Next >>",fg="black",command=self.next_matching,disabledforeground="black")
        self.prev_button=Button(self.root,text="<< Previous",fg="black",command=self.prev_matching,disabledforeground="black",
        state=DISABLED)
        self.less_button=Button(self.root,text="Less Boxes",fg="black",command=self.less_boxes)
        self.refine_button=Button(self.root,text="Resize Boxes",fg="black",command=self.less_boxes)
        self.finer_resize_button=Button(self.root,text="Finer Resize Boxes",fg="black",command=self.finer_less_boxes,padx=100)
        self.more_button=Button(self.root,text="More Boxes",fg="black",command=self.more_boxes)
        self.save_button=Button(self.root,text="Save JSON",fg="black",command=self.save,disabledforeground="black",padx=120)
        
        self.start_button.place(x=1150,y=100)
        self.next_button.place(x=1270,y=100)
        self.prev_button.place(x=1050,y=100)
        self.less_button.place(x=1050,y=150)
        self.refine_button.place(x=1150,y=150)
        self.more_button.place(x=1280,y=150)
        self.finer_resize_button.place(x=1050,y=200)
        self.save_button.place(x=1050,y=300)

    def open_json_file(self):
        self.root.filename=filedialog.askopenfilename(initialdir=CUR_DIR,
        title="Select A File",filetypes=(("JSON Files","*.json"),
        ("All Files","*.*")))
        self.tm=TemplateMatcher(self.root.filename)
        # JSON data\
        self.jsondata=self.tm.labelmeData
        # ImagePath
        self.imagePath=self.tm.imagePath
        # All labels
        self.all_labels=list(self.tm.data['label'].unique())
        self.all_labels=sorted([int(i) for i in self.all_labels])
        self.all_labels=[str(i) for i in self.all_labels]
        # All the images are saved for every labels 
        self.all_images=[]
        # All the boxes are saved for every labels
        self.all_boxes=dict()
        for lb in self.all_labels:
            self.all_boxes[lb]=[]
        
        # All the metadata about the box
        self.all_box_dict=dict()
        for lb in self.all_box_dict:
            self.all_box_dict[lb]=[]
        # All the thresholds are saved for every labels
        self.all_threshold=dict()
        for lb in self.all_labels:
            self.all_threshold[lb]=0.45
        
        # Present Label
        self.label_no=0


        # Plot the image and the button.
        image=Image.fromarray(self.tm.original_image.copy())
        image=image.resize((1000,1000),Image.ANTIALIAS)
        self.frame(image)
        self.button()
        self.next_button['state']=DISABLED

        #self.matching_window()
        
       
if __name__=="__main__":
    gui=GUI()
    gui.root.mainloop()

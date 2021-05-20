from alignment import alignment
from tkinter import *
import numpy as np
from tkinter.constants import DISABLED, NORMAL
import tkinter as tk
from tkinter.filedialog import askopenfilenames
from platform import system
import cv2
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk,Image,ImageDraw
from templatematcher import TemplateMatcher
from templatesaver import TemplateSaver
import os
import shutil
import random
import sys
import json
import webbrowser
import warnings
import colorsys
platformD = system()
if platformD == 'Darwin':
    from tkmacosx import Button


class ransac:

    def __init__(self):
        self.root=Tk()

        self.root.title("Ransac Image Alignment")
        self.root.geometry("400x400")

        # buttons
        self.button()

    
    def button(self):
        self.choose_folder=Button(self.root,text="Choose Folder",fg="black",command=self.open_folder,disabledforeground="black")
        self.open_source=Button(self.root,text="Choose Source",fg="black",command=self.open_source,disabledforeground="black")
        self.open_target=Button(self.root,text="Choose Target",fg="black",command=self.open_target,disabledforeground="black")
        self.start=Button(self.root,text="Start Alignment",fg="black",command=self.alignment)

        self.choose_folder.place(x=10,y=10)
        self.open_source.place(x=10,y=90)
        self.open_target.place(x=10,y=50)
        self.start.place(x=10,y=130)
    
    def open_folder(self):
        self.folder=filedialog.askdirectory(title="Select A Folder")
        self.all_images=os.listdir(self.folder)
        self.target=self.folder+"/"+self.all_images[-1]
        self.sources=[self.folder+"/"+i for i in self.all_images[:-1]]
    
    def open_source(self):
        self.sources=filedialog.askopenfilenames(title="Select Source Images",filetypes=(("jpg Files","*.jpg"),("All Files","*.*")))

    def open_target(self):
        self.target=filedialog.askopenfilename(title="Select Target Image",filetypes=(("jpg Files","*.jpg"),("All Files","*.*")))
    
    def alignment(self):
        self.target_image=np.asarray(Image.open(self.target),dtype=np.uint8)        
        for Is in self.sources:
            final_image=alignment(Is,self.target)
            name="/Users/ryzenx/other/"+It.split("/")[-1]
            cv2.imwrite(name,final_image)
        target_name="/Users/ryzenx/other/"+It.split("/")[-1].split(".")[0]+"_ref.jpg"
        cv2.imwrite(target_name,self.target_image)

if __name__=="__main__":
    root = ransac()
    root.root.mainloop()

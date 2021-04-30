from tkinter import *
from tkmacosx import Button
from tkinter import messagebox
from tkinter import filedialog
import os
CURDIR=os.curdir


CUR_DIR=os.curdir

class GUI():

    def __init__(self):
        # Create a gui
        self.root=Tk()
        self.root.title("Auto Image Annotator")
        self.root.geometry("800x800")

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
        self.file_menu.add_command(label="Exit",command=self.root.quit)

        # Edit Menu
        self.edit_menu=Menu(self.my_menu)
        self.my_menu.add_cascade(label="Edit",menu=self.edit_menu)
        self.edit_menu.add_command(label="Undo",state=DISABLED)

        self.root.config(menu=self.my_menu)
    
    def frame(self):
        self.image_frame=Frame(self.root,width=400,height=400,bg="red")
        self.image_frame.pack(fill="both",expand=1)

    def restart(self):
        self.image_frame.pack_forget()


    def open_json_file(self):
        self.root.filename=filedialog.askopenfilename(initialdir=CUR_DIR,
        title="Select A File",filetypes=(("JSON Files","*.json"),
        ("All Files","*.*")))

        self.frame()

        


    
if __name__=="__main__":
    gui=GUI()
    gui.root.mainloop()

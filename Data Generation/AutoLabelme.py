from tkinter import *
from tkmacosx import Button
from tkinter import messagebox
from tkinter import filedialog
import os



CUR_DIR=os.curdir

class GUI():

    def __init__(self):
        # Create a gui
        self.root=Tk()

        self.root.title("Auto Image Annotator")

        # Create two Buttons
        self.all_buttons()

        self.all_grids()

        
    # All the commands

    def open_file(self,json=False):
        """ Open File"""
        if json:
            self.json_file=filedialog.askopenfilename(initialdir=CUR_DIR,
            title="Select the Appropriate JSON File",filetypes=(("json files","*.json"),("all files","*.*")))

        else:
            self.image_file=filedialog.askopenfilename(initialdir=CUR_DIR, 
            title="Select the Appropriate JSON File",filetypes=(("JSON files","*.json")))

        if self.json_file:
            self.next_button['state']= NORMAL
            self.next_button.grid(row=1,column=1)


    def working_window(self):
        """ Open the Working Window. Main work will be done here."""
        self.work=Toplevel()

    def ask(self):
        """ Asks to exit or not"""
        message=messagebox.askquestion("Exiting Application","Do you want to exit the application?")
        if message=="yes":
            self.root.quit()

    def close_window(self):
        """ Destroys the child windows"""
        try:
            # Destroy all child windows
            self.work.destroy()
            self.top.destroy()
        except:
            pass
        self.next_button['state']= DISABLED
        self.next_button.grid(row=1,column=1)

    def all_buttons(self):

        """ Create all the buttons"""
        self.json_button=Button(self.root,text="Open JSON",padx=2.5,fg="black",command=lambda: self.open_file(True))
        #self.image_button=Button(self.root,text="Open Image",fg="black",command=self.open_file)
        self.exit_button=Button(self.root,text="Exit",padx=10,fg="black",command=self.ask)
        self.next_button=Button(self.root,text="Next",padx=9,fg="black",disabledbackground="lightgray",
        disabledforeground='white',state=DISABLED,command=self.working_window)
        self.restart_button=Button(self.root,text="Restart",padx=10,fg="black",command=self.close_window)

    def all_grids(self):
        """ Place the buttons on the GUI"""
        self.json_button.grid(row=0,column=0)
        self.restart_button.grid(row=1,column=0)
        #self.image_button.grid(row=1,column=0)
        self.exit_button.grid(row=0,column=1)
        self.next_button.grid(row=1,column=1)
        



    
if __name__=="__main__":
    gui=GUI()
    gui.root.mainloop()
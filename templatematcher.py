import numpy as np
import pandas as pd
from skimage.feature import match_template

class TemplateMatcher:

    """ Template Matching """

    def __init__(self,data,label_name,num_object=1,slack=0):

        """
        Params:

        data --> PANDAS.DATAFRAME. Dataset containing the template information
        label_name --> INT or STRING. Label_name for which Template Matching will be used
        num_object --> INT. Number of objects to be tracked.
        slack --> INT. Variable to adjust the bounding box

        """

        self.data=data
        self.label_name=label_name            
        self.num_object=num_object
        self.slack=slack

        # If label index is provided, we need the label name.
        if type(self.label_index)==int:
            self.label_name=self.data['label'].unique[self.label_name]


    
    def find_template(self,image,template):
        """ Find the xy coordinates for the template in image"""

        result=match_template(image,template)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]

        return x,y
    
    def find_all_template(self,image,template):
        """ Find all the xy coordinates for the template in image """

        bbox=self.data[self.data['label']=self.label_name]['bbox']
        x_axis=[bbox[0][0],bbox[1][0]]
        y_axis=[bbox[1][0],bbox[1][1]]

        height=y_axis[1]-y_axis[1]+self.slack
        width=x_axis[1]-x_axis[0]-196+self.slack
        i=0
        boxes=[]
    
    def plot_image(self,image,boxes=[])
        
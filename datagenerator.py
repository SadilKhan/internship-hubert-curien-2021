import numpy as np
import os
import numpy as np
import pandas as pd
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import PIL
from PIL import ImageOps,Image



class DataGenerator:
    def __init__(self):
        pass    
    def fit_transform(self,image,data,label=None):
        self.image= image
        self.data=data
        self.label=label

        gray=self.rgb2gray(self.image)
        
        # Creating cropped images of the objects
        self.data['image_array']=self.data['bbox'].apply(lambda x: np.array(self.resize_image(self.crop_image(gray,x),(256,256)),dtype=np.float32))
        self.x_train,self.y_train=self.data.iloc[[0,1,2]]['image_array'],self.data.iloc[[0,1,2]]['label']
        #self.x_test,self.y_test=self.data.iloc[:13]['image_array'],self.data.iloc[:13]['label']

        self.num_labels=self.y_train.nunique()


        for j in range(len(self.x_train)):
            # Crop the portion of image containing the bounding box
            cr_image=self.x_train.iloc[j]

            # Rotation
            for k in [45,90,135,180,225,270]:
                r_cr_image=np.array(self.rotate_image(cr_image,k))
                self.x_train=self.x_train.append(pd.Series([r_cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))
            
            # Flip Image
            vf_cr_image=np.array(self.flip_image(cr_image))
            self.x_train=self.x_train.append(pd.Series([r_cr_image]))
            self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))

            # Mirror Image
            mf_cr_image=np.array(self.mirror_image(cr_image))
            self.x_train=self.x_train.append(pd.Series([mf_cr_image]))
            self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))
            
            # Affine Transformation

            for p in np.array(range(1,10))/10:
                af_cr_image=np.array(self.affine_transform(cr_image,(1,p,1,0,1,0)))
                self.x_train=self.x_train.append(pd.Series([af_cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))

                af_cr_image=np.array(self.affine_transform(cr_image,(1,-p,1,0,1,0)))
                self.x_train=self.x_train.append(pd.Series([af_cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))

                af_cr_image=np.array(self.affine_transform(cr_image,(1,0,1,p,1,0)))
                self.x_train=self.x_train.append(pd.Series([af_cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))

                af_cr_image=np.array(self.affine_transform(cr_image,(1,0,1,-p,1,0)))
                self.x_train=self.x_train.append(pd.Series([af_cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))
            for p in range(10):
                self.x_train=self.x_train.append(pd.Series([cr_image]))
                self.y_train=self.y_train.append(pd.Series(self.y_train.iloc[j]))

        # Scaling Image

        self.x_train=self.x_train.apply(lambda x: self.scale_image(x))
        #self.x_test=self.x_test.apply(lambda x: self.scale_image(x))

        # Transforming the arrays to multi-dimensional
        self.x_train=np.vstack(self.x_train).reshape(len(self.x_train),256,256)
        #self.x_test=np.vstack(self.x_test).reshape(len(self.x_test),256,256)

        self.x_train=np.expand_dims(self.x_train,-1)
        #self.x_test=np.expand_dims(self.x_test,-1)


        self.y_train=np.array(self.y_train,dtype=np.int32)
        #self.y_train=to_categorical(self.y_train,self.num_labels)

        #self.y_test=np.array(self.y_test,dtype=np.int32)
        #self.y_test=to_categorical(self.y_test,self.num_labels)
        #self.x_test,self.y_test

        return self.x_train,self.y_train


    def rgb2gray(self,image):
        """ Transform to Grayscale """
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def plot_image(self,image):
        """Plotting Image. Return PIL image"""
        return Image.fromarray(np.uint8(image))

    def flaten_image(self,array):
        """ Flatten a numpy array """
        return array.flatten()

    def scale_image(self,array):
        """ Scale an array from 0-255 to 0-1 range"""
        return array/255

    def crop_image(self,image,x):
        """Cropping Image"""
        return image[x[0][1]:x[1][1],x[0][0]:x[1][0]]

    def rotate_image(self,image,angle):
        """Rotation"""
        if type(image)==np.ndarray:
            image=self.plot_image(np.uint8(image))
        return image.rotate(angle)

    def resize_image(self,image,new_size):
        """Resize Image"""
        if type(image)==np.ndarray:
            image=self.plot_image(np.uint8(image))
        return image.resize(new_size)

    def flip_image(self,image):
        """Vertical Flip"""
        if type(image)==np.ndarray:
            image=self.plot_image(np.uint8(image))
        return ImageOps.flip(image)

    def mirror_image(self,image):
        """Horizontal Flip"""
        if type(image)==np.ndarray:
            image=self.plot_image(np.uint8(image))
        return ImageOps.mirror(image)

    def affine_transform(self,image,affine_matrix):
        """ Affine Transformation """
        if type(image)==np.ndarray:
            image=self.plot_image(np.uint8(image))
        shape=image.size
        return image.transform(shape,Image.AFFINE,affine_matrix,resample=Image.BILINEAR)
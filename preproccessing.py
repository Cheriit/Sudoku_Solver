#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import imshow
import skimage.io as io
from pylab import *
import skimage
from skimage import data, filters, exposure, feature
from skimage.filters import rank
from skimage.util.dtype import convert
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from matplotlib import pylab as plt  
import numpy as np 
from numpy import array
from IPython.display import display
from ipywidgets import interact, interactive, fixed
from IPython.core.display import clear_output
from skimage import measure
import cv2


# In[2]:


def get_area(rect):
    x,y,w,h = rect
    return(w*h)


# In[3]:


def inverse(image):
    return (255-image)   


# In[18]:


def clean_image(img, black_and_white):
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_clean = cv2.fastNlMeansDenoising(img,h=5) 
    avr = average(img_clean)
    sd = std(img_clean)
    if avr>200:
        ret,img = cv2.threshold(img_clean, avr, 255,cv2.THRESH_BINARY_INV)
    elif avr>160:
        ret,img = cv2.threshold(img_clean, avr-2.1*sd, 255,cv2.THRESH_BINARY_INV)
    else:
        ret,img = cv2.threshold(img_clean, avr-1.5*sd, 255,cv2.THRESH_BINARY_INV)

    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(cnt) for cnt in contours]
    rects = sorted(rects,key=get_area,reverse=True)
    x,y,w,h = rects[0]
    img_to_show = img_clean[y:y+h,x:x+w]
    if not black_and_white:
        return img_to_show
    avr = average(img_to_show)
    sd = std(img_to_show)
    if avr>200:
        ret,img = cv2.threshold(img_to_show, avr, 255,cv2.THRESH_BINARY_INV)
    elif avr>160:
        img_to_show = inverse(img_to_show)
        img_to_show = cv2.dilate(img_to_show, np.ones((3,3), np.uint8) , iterations=3) 
        img_to_show = inverse(img_to_show)
        ret,img = cv2.threshold(img_to_show, avr-1.5*sd, 255,cv2.THRESH_BINARY_INV)
    else:
        img_to_show = inverse(img_to_show)
        img_to_show = cv2.dilate(img_to_show, np.ones((5,5), np.uint8) , iterations=3) 
        img_to_show = inverse(img_to_show)
        ret,img = cv2.threshold(img_to_show, avr-1.3*sd, 255,cv2.THRESH_BINARY_INV)
    print("Średnia: "+str(avr)+" Odchylenie:"+str(sd))
    return img


# In[7]:


def clean_image_to_file(img, black_and_white,path):
    cv2.imwrite(path, clean_image(img,black_and_white))


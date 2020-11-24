#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[10]:


def get_area(rect):
    x,y,w,h = rect
    return(w*h)


# In[11]:


def inverse(image):
    return (255-image)   


# In[13]:


def clean_image(img):
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
    #if not black_and_white:
   #     return img_to_show
    avr = average(img_to_show)
    sd = std(img_to_show)
    if avr>200:
        ret,img = cv2.threshold(img_to_show, avr, 255,cv2.THRESH_BINARY_INV)
    elif avr>160:
        img = inverse(img_to_show)
        img = cv2.dilate(img, np.ones((3,3), np.uint8) , iterations=3) 
        img = inverse(img)
        ret,img = cv2.threshold(img, avr-1.5*sd, 255,cv2.THRESH_BINARY_INV)
    else:
        img = inverse(img_to_show)
        img = cv2.dilate(img, np.ones((5,5), np.uint8) , iterations=3) 
        img = inverse(img)
        ret,img = cv2.threshold(img, avr-1.3*sd, 255,cv2.THRESH_BINARY_INV)
    print("Średnia: "+str(avr)+" Odchylenie:"+str(sd))
    return img_to_show, img


# In[5]:


def clean_image_to_file(img, black_and_white,path):
    grayscale, thresholded = clean_image(img)
    if black_and_white:
        cv2.imwrite(path, thresholded)
    else:
        cv2.imwrite(path, grayscale)


# In[18]:


a,b = clean_image(cv2.imread("img/easy7.jpg"))
plt.imshow(a, cmap='gray')
plt.show()
plt.imshow(b, cmap='gray')
plt.show()


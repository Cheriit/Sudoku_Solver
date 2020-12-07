#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from math import floor, ceil
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage import util

from helpers import rescale_img
import helpers

model = None
images_for_nr=[]
images_for_nn=[]

def predict(cut_digit_img: np.ndarray) -> int:
    global model
    global images_for_nn
    global images_for_nr
    images_for_nr.append(rescale_img(cut_digit_img,28))
    minmax = (cut_digit_img.flatten().min(), cut_digit_img.flatten().max())

    cut_digit_img = rescale_intensity(cut_digit_img, minmax)
    cut_digit_img = util.invert(cut_digit_img)
    cut_digit_img = cv2.erode(cut_digit_img, np.ones((3, 1), np.uint8), iterations=1)
    cut_digit_img = rescale_img(cut_digit_img, 18)
    if len(cut_digit_img[0])>28:
        return 0
        helpers.wait_for_key_on_value_error("digit wider then its height! maybe board detection problem?")


    #_, cut_digit_img = cv2.threshold(cut_digit_img, 110, 255, cv2.THRESH_BINARY) # Possible deletion

    # cut_digit_img = Image.fromarray(np.uint8(cut_digit_img))

    if model is None:
        # model = load_model("number_recognition_model/number_recognition_model_conv.h5") # Conv model
        model = load_model("number_recognition_model/number_recognition_model.h5") # Dense model

    image_for_nn = process_img(cut_digit_img)
    img_for_show=image_for_nn.reshape(28,28)
    images_for_nn.append(img_for_show)
    res = model.predict([image_for_nn])[0]
    return np.argmax(res)


def process_img(cut_digit_img: np.ndarray) -> np.ndarray:
    # cut_digit_img = cut_digit_img.convert('L')
    # cut_digit_img = np.array(cut_digit_img)

    cut_digit_img = cut_digit_img / 255
    new_image = np.zeros((28, 28))
    old_image_rows, old_image_cols = cut_digit_img.shape

    new_image[
        14 - floor(old_image_rows / 2):14 + ceil(old_image_rows / 2),
        14 - floor(old_image_cols / 2):14 + ceil(old_image_cols / 2)
    ] = cut_digit_img

    # new_image = new_image.reshape(1, 28, 28, 1) # Conv model
    new_image = new_image.reshape(1, 784) # Dense model
    new_image[new_image < 0.2] = 0
    return new_image


def show_imgs_for_nn():
    if(len(images_for_nn)>0):
        cv2.imshow('imgsForNN',np.hstack(images_for_nn))

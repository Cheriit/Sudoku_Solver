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

model = None


def predict(cut_digit_img: np.ndarray) -> int:
    global model

    minmax = (cut_digit_img.flatten().min(), cut_digit_img.flatten().max())
    cut_digit_img = rescale_intensity(cut_digit_img, minmax)
    cut_digit_img = util.invert(cut_digit_img)
    cut_digit_img = rescale_img(cut_digit_img, 24)
    _, cut_digit_img = cv2.threshold(cut_digit_img, 110, 255, cv2.THRESH_BINARY) # Possible deletion

    # cut_digit_img = Image.fromarray(np.uint8(cut_digit_img))

    if model is None:
        model = load_model("number_recognition_model/number_recognition_model.h5")

    image_for_nn = process_img(cut_digit_img)
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

    new_image = new_image.reshape(1, 28, 28, 1)
    new_image[new_image < 0.2] = 0
    return new_image

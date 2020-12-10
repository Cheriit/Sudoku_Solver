#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np


def rescale_img(img: np.ndarray, wanted_y: int = 1024) -> np.ndarray:
    dimensions = img.shape
    target_x = max(1,int(dimensions[0] * wanted_y / dimensions[0]))
    target_y = max(1,int(dimensions[1] * wanted_y / dimensions[0]))
    if dimensions[0]>0 and dimensions[1]>0:
        img = cv2.resize(img, (target_y, target_x))
    return img


def get_area(rect: tuple) -> float:
    x, y, w, h = rect
    return w * h


def inverse(image: np.ndarray) -> np.ndarray:
    return 255 - image


# snippet used from:
# https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
def order_image_points(four_points):
    rect = np.zeros((4, 2), dtype="float32")

    s = four_points.sum(axis=1)
    rect[0] = four_points[np.argmin(s)]
    rect[2] = four_points[np.argmax(s)]

    diff = np.diff(four_points, axis=1)
    rect[1] = four_points[np.argmin(diff)]
    rect[3] = four_points[np.argmax(diff)]

    return rect


def wait_for_window_close_or_keypress(exit_after_wait=True, window='drawOutput'):
    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) > 0:
        keycode = cv2.waitKey(1)
        if keycode >= 0:
            break
    if exit_after_wait:
        exit(0)


def wait_for_key_on_value_error(message):
    print("ValueError" + message)
    wait_for_window_close_or_keypress(exit_after_wait=False, window='findBoard')
    raise (ValueError(message))


def load_img(name, useAbsPath=False):
    if useAbsPath:
        original_img = cv2.imread(name, cv2.IMREAD_COLOR)
    else:
        original_img = cv2.imread('img/' + name, cv2.IMREAD_COLOR)
    original_img = rescale_img(original_img, 800)
    return original_img


def save_img(save_name, stage, img):
    cv2.imwrite('test_img/' + save_name[:-4] + stage + '.jpg', img)


def reshape81to9x9(arr):
    reshaped = []
    for i in range(9):
        reshaped.append(arr[i * 9:(i + 1) * 9])
    return reshaped


def many_fields_to_one_img(fields):
    reshaped = reshape81to9x9(fields)
    return np.vstack([np.hstack(row) for row in reshaped])

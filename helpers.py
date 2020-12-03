#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np


def rescale_img(img: np.ndarray, wanted_y: int = 1024) -> np.ndarray:
    dimensions = img.shape
    target_x = int(dimensions[0] * wanted_y / dimensions[0])
    target_y = int(dimensions[1] * wanted_y / dimensions[0])
    img = cv2.resize(img, (target_y, target_x))
    return img


def get_area(rect: tuple) -> float:
    x, y, w, h = rect
    return w * h


def inverse(image: np.ndarray) -> np.ndarray:
    return 255 - image


def order_image_points(four_points):
    rect = np.zeros((4, 2), dtype="float32")

    s = four_points.sum(axis=1)
    rect[0] = four_points[np.argmin(s)]
    rect[2] = four_points[np.argmax(s)]

    diff = np.diff(four_points, axis=1)
    rect[1] = four_points[np.argmin(diff)]
    rect[3] = four_points[np.argmax(diff)]

    return rect


def wait_for_window_close_or_keypress():
    while cv2.getWindowProperty('drawOutput', cv2.WND_PROP_VISIBLE) > 0:
        keyCode = cv2.waitKey(1)
        if keyCode>=0:
            break
    exit(0)
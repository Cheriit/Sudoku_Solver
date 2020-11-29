#!/usr/bin/env python
# coding: utf-8

import cv2
import imutils
import numpy as np
import number_recognition
from helpers import order_image_points, inverse, rescale_img

debug = False


def warp_perspective(img: np.ndarray, board_contour: np.ndarray) -> np.ndarray:
    pts = board_contour.reshape(4, 2)
    rect = order_image_points(pts)

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp


def detect_board(thresholded_img: np.ndarray, img: np.ndarray) -> np.ndarray:
    original_for_warp = img.copy()

    # find contours in the thresholded image, keep only the 10 largest contours
    found_contours = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_contours = imutils.grab_contours(found_contours)
    found_contours = sorted(found_contours, key=cv2.contourArea, reverse=True)[:10]

    # find largest 4-sided contour
    board_contour = None
    epsilon = 0.015
    for cont in found_contours:
        arclen = cv2.arcLength(cont, True)
        approximated_polygon = cv2.approxPolyDP(cont, epsilon * arclen, True)
        if len(approximated_polygon) == 4:
            board_contour = approximated_polygon
            break

    # drawing the largest detected in red (doesn't have to be 4-sided)
    if board_contour is None or debug:
        cv2.drawContours(img, [found_contours[0]], -1, (0, 0, 255), 3)

    # draw found board_contour in green
    if board_contour is None:
        thresholded_color_temp = cv2.cvtColor(thresholded_img, cv2.COLOR_GRAY2BGR)
        to_show = np.hstack((thresholded_color_temp, img))
        cv2.imshow("finding board", rescale_img(to_show))
        raise ValueError("Can not find board on image")

    warped = warp_perspective(original_for_warp, board_contour)
    return warped


def cut_out_fields(warped: np.ndarray) -> np.ndarray:
    y, x, _ = warped.shape
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sudoku_field_img_array = []

    for row in range(9):
        sudoku_field_img_array.append([])
        for column in range(9):
            x_min = column * x // 9
            y_min = row * y // 9 + 1
            x_max = (column + 1) * x // 9 - 1
            y_max = (row + 1) * y // 9 - 1
            field_img = warped_gray[y_min:y_max, x_min:x_max]
            sudoku_field_img_array[-1].append(field_img)

            if debug:
                # draw yellow squares for each field (just visualization) and their centers as red circles
                warped = cv2.rectangle(
                    warped, (column * x // 9, row * y // 9), ((column + 1) * x // 9, (row + 1) * y // 9), (0, 255, 255),
                    1)
                warped = cv2.circle(warped, (x // 18 + column * x // 9, y // 18 + row * y // 9), radius=3,
                                    color=(0, 0, 255), thickness=1)
    if debug:
        cv2.imshow('cut board', rescale_img(warped))
    return sudoku_field_img_array


def process_fields(sudoku_field_img_array: np.ndarray) -> np.ndarray:
    recognized_fields = []
    for row_id in range(len(sudoku_field_img_array)):
        for col_id in range(len(sudoku_field_img_array[row_id])):
            original_img = sudoku_field_img_array[row_id][col_id]
            thresholded_img = threshold_field_image(original_img.copy())
            dim = thresholded_img.shape

            contours, _ = cv2.findContours(thresholded_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found = None
            if len(contours) != 0:
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    # if the contour is sufficiently large, it must be a digit
                    # height and width limiters to eliminate grid lines detection
                    if (dim[1] * 5 // 28 < w < dim[1] * 25 // 28 and dim[1] * 1 // 28 <= x <= dim[
                        1] * 27 // 28) and \
                            (dim[0] * 10 // 28 < h < dim[0] * 25 // 28 and dim[0] * 1 // 28 <= y <= dim[
                                0] * 27 // 28):
                        found = (x, y, w, h)
                        break

            if found is None:
                recognized_fields.append(0)
            else:
                (x, y, w, h) = found
                cut_digit = original_img[y:y + h, x:x + w]
                digit = number_recognition.predict(cut_digit)
                recognized_fields.append(digit)

    return np.array(recognized_fields).reshape(9, 9)


def cut_image(thresholded_img: np.ndarray, original_img: np.ndarray, rescale: bool = False, enable_debug: bool = False) -> np.ndarray:
    global debug
    debug = enable_debug

    if rescale:
        thresholded_img = rescale_img(thresholded_img)
        original_img = rescale_img(original_img)

    warped = detect_board(thresholded_img, original_img)
    return cut_out_fields(warped)


def threshold_field_image(img: np.ndarray) -> np.ndarray:
    img = cv2.fastNlMeansDenoising(img, h=5)
    avr = np.average(img)
    sd = np.std(img)
    if avr > 200:
        ret, img = cv2.threshold(img, avr, 255, cv2.THRESH_BINARY_INV)
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    elif avr > 160:
        ret, img = cv2.threshold(img, avr - 1.5 * sd, 255, cv2.THRESH_BINARY_INV)
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    else:
        ret, img = cv2.threshold(img, avr - 1.3 * sd, 255, cv2.THRESH_BINARY_INV)
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
    return img


def threshold_board_image(img: np.ndarray) -> np.ndarray:
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_clean = cv2.fastNlMeansDenoising(img, h=5)

    avr = np.average(img_clean)
    sd = np.std(img_clean)

    if avr > 200:
        ret, img = cv2.threshold(img_clean, avr, 255, cv2.THRESH_BINARY_INV)
    elif avr > 160:
        img = inverse(img_clean)
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations=3)
        img = inverse(img)
        ret, img = cv2.threshold(img, avr - 1.5 * sd, 255, cv2.THRESH_BINARY_INV)
    else:
        img = inverse(img_clean)
        img = cv2.dilate(img, np.ones((5, 5), np.uint8), iterations=3)
        img = inverse(img)
        ret, img = cv2.threshold(img, avr - 1.3 * sd, 255, cv2.THRESH_BINARY_INV)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=60)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), 255, 6)
    return img


def draw_output(output_array: np.ndarray) -> np.ndarray:
    pass
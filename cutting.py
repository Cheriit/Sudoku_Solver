import cv2
import os
import numpy as np
import imutils
from skimage.exposure import rescale_intensity
from skimage import util

debug = True


def rescalle_img(img):
    dimensions = img.shape
    wanted_x = 800
    target_x = int(dimensions[0] * wanted_x / dimensions[0])
    target_y = int(dimensions[1] * wanted_x / dimensions[0])
    img = cv2.resize(img, (target_y, target_x))
    return img


def order_points(four_points):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = four_points.sum(axis=1)
    rect[0] = four_points[np.argmin(s)]
    rect[2] = four_points[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(four_points, axis=1)
    rect[1] = four_points[np.argmin(diff)]
    rect[3] = four_points[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def warp(img, board_contour):
    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = board_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # multiply the rectangle by the original ratio
    # rect *= ratio
    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp


def find_board_and_warp_it(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_for_warp = img.copy()
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=60)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(edges, (x1, y1), (x2, y2), 255, 6)

    # find contours in the edged image, keep only the 10 largest contours
    found_contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    found_contours = imutils.grab_contours(found_contours)
    found_contours = sorted(found_contours, key=cv2.contourArea, reverse=True)[:10]

    # find largest 4-sided contour
    board_contour = None
    for cont in found_contours:
        arcl = cv2.arcLength(cont, True)
        epsilon = 0.015
        approx = cv2.approxPolyDP(cont, epsilon * arcl, True)
        if len(approx) == 4:
            board_contour = approx
            break

    # draw found board_contour
    if board_contour is not None and debug:
        cv2.drawContours(img, [board_contour], -1, (0, 255, 255), 3)
    # drawing the largest detected in yellow (doesn't have to be 4-sided)
    cv2.drawContours(img, [found_contours[0]], -1, (255, 0, 0), 3)
    edges_c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if debug:
        cv2.imshow("preprocess", np.hstack((edges_c, img)))

    if board_contour is None:
        return None
    else:
        warped = warp(original_for_warp, board_contour)
        return warped


def cut_board(warped):
    y, x, _ = warped.shape
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sudoku_board = []
    for row in range(9):
        sudoku_board.append([])
        for column in range(9):
            xmin = column * x // 9
            ymin = row * y // 9 + 1
            xmax = (column + 1) * x // 9 - 1
            ymax = (row + 1) * y // 9 - 1
            field_img = warped_gray[ymin:ymax, xmin:xmax]
            sudoku_board[-1].append(field_img)
            if debug:
                # draw yellow squares for each field (just visualization) and their centers as red circles
                warped = cv2.rectangle(
                    warped, (column * x // 9, row * y // 9), ((column + 1) * x // 9, (row + 1) * y // 9), (0, 255, 255),
                    1)
                warped = cv2.circle(warped, (x // 18 + column * x // 9, y // 18 + row * y // 9), radius=3,
                                    color=(0, 0, 255), thickness=1)
    if debug:
        cv2.imshow('cut board', warped)
    return sudoku_board


def process_fields(sudoku):
    recognized_fields = []
    if debug:
        digit_imgs = []
    for row_id in range(len(sudoku)):
        for col_id in range(len(sudoku[row_id])):
            # dim=(28,28)
            # img = cv2.resize(sudoku[row_id][col_id], dim)
            img = sudoku[row_id][col_id]
            dim = img.shape
            _, thresholded_small = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresholded_small, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            found = None
            if len(contours) != 0:
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    # if the contour is sufficiently large, it must be a digit
                    # height and width limiters to eliminate grid lines detection
                    if (w > dim[1] * 5 // 28 and w < dim[1] * 25 // 28 and x >= dim[1] * 1 // 28 and x <= dim[
                        1] * 27 // 28) and \
                            (h > dim[0] * 10 // 28 and h < dim[0] * 25 // 28 and y >= dim[0] * 1 // 28 and y <= dim[
                                0] * 27 // 28):
                        found = (x, y, w, h)
                        break
            if found is None:
                recognized_fields.append(None)
                if debug:
                    digit_img = np.zeros(dim, dtype=np.uint8)
                    digit_imgs.append(digit_img)
            else:
                (x, y, w, h) = found
                cut_digit = img[y:y + h, x:x + w]
                minmax = (cut_digit.flatten().min(), cut_digit.flatten().max())
                cut_digit = rescale_intensity(cut_digit, minmax)
                # nnimg = np.zeros((28, 28), dtype=np.uint8)
                # nnimg[y:y + h, x:x + w] = util.invert(cut_digit)
                # nnimg = util.invert(nnimg)
                # nnimg = cv2.dilate(nnimg, np.ones((1, 1), np.uint8), iterations=20)
                # nnimg_4d = nnimg.reshape(1, 28, 28, 1)
                # cv2.imshow('cut digit', cut_digit )
                # cv2.waitKey(0)
                if debug:
                    digit_img = np.zeros(dim, dtype=np.uint8)
                    digit_img[y:y + h, x:x + w] = util.invert(cut_digit)
                    digit_imgs.append(digit_img)
                # here should be number recognition for each digit
                # ============
                digit = 1
                # ============
                recognized_fields.append(digit)
    if debug:
        digit_imgs = np.array(digit_imgs, dtype=object).reshape(9, 9)
        cv2.imshow('fields for nn', np.vstack([np.hstack(row) for row in digit_imgs]))
    return np.array(recognized_fields).reshape(9, 9)


def run_cutting(img, rescalle=True):
    if rescalle:
        img = rescalle_img(img)
    warped = find_board_and_warp_it(img)
    if warped is None:
        print("board not found")
        return None
    else:
        sudoku = cut_board(warped)
        return process_fields(sudoku)


def main():
    img = cv2.imread('img/medium2.jpg', cv2.IMREAD_COLOR)
    print(run_cutting(img))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

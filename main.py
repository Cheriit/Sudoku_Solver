#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
import solvers.backtracking as backtracking
import time



def test_solver(solver_type: str) -> None:
    # Array for testing
    output_array = [
        [0, 4, 0, 2, 0, 1, 0, 6, 0]
        , [0, 0, 0, 0, 0, 0, 0, 0, 0]
        , [9, 0, 5, 0, 0, 0, 3, 0, 7]
        , [0, 0, 0, 0, 0, 0, 0, 0, 0]
        , [5, 0, 7, 0, 8, 0, 1, 0, 4]
        , [0, 1, 0, 0, 0, 0, 0, 9, 0]
        , [0, 0, 1, 0, 0, 0, 6, 0, 0]
        , [0, 0, 0, 7, 0, 5, 0, 0, 0]
        , [6, 0, 8, 9, 0, 4, 5, 0, 3]
    ]

    start_time = time.time()

    # Solve the sudoku
    if solver_type == "basic":
        alg = backtracking.Basic(np.array(output_array))
    elif solver_type == "possibilities":
        alg = backtracking.Possible(np.array(output_array))
    else:
        raise NameError("Selecter solver doesn't exists")

    print('Did solver succeed in solving the sudoku: {}'.format(alg.solve()))
    print('Time spent on solving: {}'.format(time.time() - start_time))
    print('Solved sudoku board:\n', alg.grid)


def solve(image_path: str) -> None:
    import cv2
    from image_processing import threshold_board_image, cut_image, process_fields, draw_output

    # loading image
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Thresholding to find the board
    thresholded = threshold_board_image(original_img)

    # Cutting the board to separate fields
    sudoku_field_img_array = cut_image(thresholded, original_img)
    if sudoku_field_img_array is None:
        raise ValueError("Board not found")

    # Find digits in thresholded images and recognize them
    output_array = process_fields(sudoku_field_img_array)

    # Solve the sudoku
    alg = backtracking.Basic(np.array(output_array))
    if not alg.solve():
        raise ValueError("Cannot solve sudoku")
    # draw the output to the original image
    draw_output(output_array)

    cv2.waitKey(0)


def test() -> None:
    import cv2
    from image_processing import threshold_board_image, cut_image, process_fields, draw_output

    start_time = time.time()

    # loading image
    original_img = cv2.imread('img/medium2.jpg', cv2.IMREAD_COLOR)


    # Thresholding to find the board
    thresholded = threshold_board_image(original_img)

    # Cutting the board to separate fields
    sudoku_field_img_array = cut_image(thresholded, original_img, enable_debug=True)
    if sudoku_field_img_array is None:
        cv2.waitKey(0)
        exit()

    # Find digits in thresholded images and recognize them
    output_array = process_fields(sudoku_field_img_array)

    print('Detected board layout: \n', output_array)

    # Solve the sudoku
    alg = backtracking.Basic(np.array(output_array))
    print('Did solver succeed in solving the sudoku: {}'.format(alg.solve()))
    print('Solved sudoku board:\n', alg.grid)
    print('Time spent on solving: {}'.format(time.time() - start_time))

    # draw the output to the original image
    draw_output(output_array)

    cv2.waitKey(0)


def test_recognition() -> None:
    import cv2
    from image_processing import threshold_board_image, cut_image, process_fields, draw_output

    # loading image
    original_img = cv2.imread('img/medium2.jpg', cv2.IMREAD_COLOR)
    start_time = time.time()

    # Thresholding to find the board
    thresholded = threshold_board_image(original_img)

    # Cutting the board to separate fields
    sudoku_field_img_array = cut_image(thresholded, original_img, enable_debug=False)
    if sudoku_field_img_array is None:
        cv2.waitKey(0)
        exit()

    # Find digits in thresholded images and recognize them
    output_array = process_fields(sudoku_field_img_array)
    print('Time spent on solving: {}'.format(time.time() - start_time))
    print('Detected board layout: \n', output_array)



if __name__ == "__main__":
    if len(sys.argv) == 2:

        if sys.argv[1] == "test_recognition":
            test_recognition()

        elif sys.argv[1] == "test":
            test()

        else:
            raise NameError("Incorrect argument name")

    elif len(sys.argv) == 3:

        if sys.argv[1] == "test_solver":
            test_solver(sys.argv[2])

        elif sys.argv[1] == "solve":
            solve(sys.argv[2])

        else:
            raise NameError("Incorrect argument name")

    else:
        raise ValueError("Incorrect argument count")
#!/usr/bin/env python
# coding: utf-8
import sys

import numpy as np
import solvers.backtracking as backtracking
import time
import cv2
from number_recognition import show_imgs_for_nn
from image_processing import threshold_board_image, cut_image, process_fields, draw_output,draw_detected
from helpers import wait_for_window_close_or_keypress, load_img


def main_test(image: str, use_abs_path=False, do_output_drawing=False, do_solving=False,
              show_detected_board=False, main_enable_debug=False):
    start_time = time.time()

    if use_abs_path:
        original_img = load_img(image, useAbsPath=True)
    else:
        original_img = load_img(image, useAbsPath=False)
    if original_img is None:
        raise ValueError("Image loading error!")

    # process original img
    sudoku_field_img_array = cut_image(original_img, enable_debug=main_enable_debug)
    if sudoku_field_img_array is None:
        raise ValueError("Board not found")

    detected_array = process_fields(sudoku_field_img_array)
    if show_detected_board:
        draw_detected(detected_array,None)
    print("Detected array:\n")
    print(np.array(detected_array))
    # sudoku solving
    solved_array = None
    if do_solving:
        alg = backtracking.Possible(np.array(detected_array))
        if not alg.solve():
            raise ValueError("Cannot solve sudoku")
        else:
            solved_array = alg.grid
    print("\nSolved array:\n")
    print(np.array(solved_array))

    # draw the output to the original image
    if do_output_drawing:
        if solved_array is None:
            solved_array = np.ones((9, 9), dtype="uint8")

        draw_output(detected_array, solved_array)
        show_imgs_for_nn()
        wait_for_window_close_or_keypress()
    operation_time = time.time() - start_time
    print('Time spent: {}'.format(operation_time))
    return operation_time

def test_solver(solver_type: str) -> None:
    # Array for testing
    detected_array = [
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
        alg = backtracking.Basic(np.array(detected_array))
    elif solver_type == "possibilities":
        alg = backtracking.Possible(np.array(detected_array))
    else:
        raise NameError("Selecter solver doesn't exists")

    print('Did solver succeed in solving the sudoku: {}'.format(alg.solve()))
    print('Time spent on solving: {}'.format(time.time() - start_time))
    print('Solved sudoku board:\n', alg.grid)


def solve(image_path: str) -> None:
    main_test(image_path, use_abs_path=True, do_output_drawing=True, do_solving=True)


def test() -> None:
    main_test("easy0.jpg", do_output_drawing=False, do_solving=True, show_detected_board=False, main_enable_debug=False)


def test_recognition() -> None:
    main_test("hard4.jpg", do_output_drawing=True, main_enable_debug=True)


def test_save_img() -> None:
    import os
    try:
        os.mkdir('test_img/')
    except FileExistsError:
        pass
    for name in os.listdir('test_img/'):
        os.remove('test_img/'+name)
    for photo in os.listdir('img/'):
        print(photo)
        original_img = load_img(photo)
        sudoku_field_img_array = cut_image(original_img, enable_debug=False, enable_save=True, saveName=photo)
        if sudoku_field_img_array is not None:
            detected_array = process_fields(sudoku_field_img_array, enable_save=True, saveName=photo)
            solved_array = np.ones((9, 9), dtype="uint8")
            draw_output(detected_array, solved_array, save_name=photo)



    exit(0)


if __name__ == "__main__":
    if len(sys.argv) == 2:

        if sys.argv[1] == "test_recognition":
            test_recognition()

        elif sys.argv[1] == "test":
            test()

        elif sys.argv[1] == "test_save_img":
            test_save_img()

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

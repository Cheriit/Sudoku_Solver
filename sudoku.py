import cv2
import numpy as np

import solvers.backtracking as backtracking


import preproccessing
import cutting


def solve_sudoku() -> None:
    # loading images
    original_img = cv2.imread('img/medium2.jpg', cv2.IMREAD_COLOR)
    # original_img = cv2.imread('img/hard2.jpg', cv2.IMREAD_COLOR)

    # Thresholding to find the board
    thresholded = preproccessing.board_threshold(original_img)

    # Cutting the board to separate fields
    sudoku_field_img_array = cutting.run_cutting(thresholded, original_img, enable_debug=False)
    if sudoku_field_img_array is None:
        cv2.waitKey(0)
        exit()

    # Find digits in thresholded images and recognize them
    output_array = cutting.process_fields(sudoku_field_img_array)
    print('Detected board layout: \n', output_array)

    # correct array for solver testing (img/medium2.jpg)
    output_array = \
        [[0, 4, 0, 2, 0, 1, 0, 6, 0]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [9, 0, 5, 0, 0, 0, 3, 0, 7]
            , [0, 0, 0, 0, 0, 0, 0, 0, 0]
            , [5, 0, 7, 0, 8, 0, 1, 0, 4]
            , [0, 1, 0, 0, 0, 0, 0, 9, 0]
            , [0, 0, 1, 0, 0, 0, 6, 0, 0]
            , [0, 0, 0, 7, 0, 5, 0, 0, 0]
            , [6, 0, 8, 9, 0, 4, 5, 0, 3]]
    # output_array = [[3, 0, 4, 0], [0, 1, 0, 2], [0, 4, 0, 3], [2, 0, 1, 0]]

    # Solve the sudoku
    # alg = backtracking.Basic(np.array(output_array))
    alg = backtracking.Possible(np.array(output_array))
    print('Did solver succeed in solving the sudoku: {}'.format(alg.solve()))
    print('Solved sudoku board:\n', alg.grid)

    # draw the output to the original image
    cutting.draw_output(output_array)

    cv2.waitKey(0)


if __name__ == "__main__":
    solve_sudoku()

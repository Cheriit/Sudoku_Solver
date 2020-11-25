import cutting
import cv2


def solve_sudoku():
    # loading images
    original_img = cv2.imread('img/medium2.jpg', cv2.IMREAD_COLOR)
    #original_img = cv2.imread('img/hard2.jpg', cv2.IMREAD_COLOR)

    # Thresholding to find the board
    thresholded = cutting.test_threshold(original_img)

    # Cutting the board to separate fields
    sudoku_field_img_array = cutting.run_cutting(thresholded, original_img, enable_debug=True)
    if sudoku_field_img_array is None:
        cv2.waitKey(0)
        exit()

    # Thresholding every field
    for row_id in range(len(sudoku_field_img_array)):
        for col_id in range(len(sudoku_field_img_array[row_id])):
            field_img = sudoku_field_img_array[row_id][col_id]
            # ==============================================
            # needs to be replaced with Piotr's thresholding function
            _, thresholded_field_img = cv2.threshold(field_img, 110, 255, cv2.THRESH_BINARY_INV)
            # ==============================================
            sudoku_field_img_array[row_id][col_id] = thresholded_field_img

    # Find digits in thresholded images and recognize them
    output_array = cutting.process_fields(sudoku_field_img_array)
    print(output_array)

    cv2.waitKey(0)

solve_sudoku()
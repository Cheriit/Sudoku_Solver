from math import sqrt, floor


def is_possible(grid, field, number):
    size = len(grid[0])
    row = field // size

    for i in range(0, size):
        if grid[row][i] == number:
            return False

    col = field % size

    for i in range(0, size):
        if grid[i][col] == number:
            return False

    segment_size = floor(sqrt(size))
    segment_start_row = row - row % segment_size
    segment_start_col = col - col % segment_size

    for i in range(segment_start_row, segment_start_row + segment_size):
        for j in range(segment_start_col, segment_start_col + segment_size):
            if grid[i][j] == number:
                return False

    return True

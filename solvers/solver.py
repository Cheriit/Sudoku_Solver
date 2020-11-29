from abc import abstractmethod

import numpy as np


class Solver:

    def __init__(self, grid: np.ndarray):
        self.size: int = int(grid.shape[0])
        self.grid: np.ndarray = grid
        self.segment_size: int = int(np.sqrt(self.size))

    @abstractmethod
    def solve(self):
        pass

    def get_field_pos(self, field: int) -> (int, int):
        return int(field // self.size), int(field % self.size)

    def get_field(self, row: int, col: int) -> int:
        return int(row * self.size + col)

    def get_segment_pos(self, row: int, col: int) -> (int, int):
        return int(row - row % self.segment_size), int(col - col % self.segment_size)

    def is_possible(self, field: int, number: int) -> bool:

        row, col = self.get_field_pos(field)
        for i in range(0, self.size):
            if self.grid[row][i] == number:
                return False

        for i in range(0, self.size):
            if self.grid[i][col] == number:
                return False

        segment_start_row, segment_start_col = self.get_segment_pos(row, col)

        for i in range(int(segment_start_row), int(segment_start_row + self.segment_size)):
            for j in range(int(segment_start_col), int(segment_start_col + self.segment_size)):
                if self.grid[i][j] == number:
                    return False

        return True



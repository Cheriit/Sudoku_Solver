from abc import abstractmethod

import numpy as np


class Solver:

    def __init__(self, grid):
        self.size = grid.shape[0]
        self.grid: np.ndarray = grid
        self.segment_size = np.floor(np.sqrt(self.size))
        self.get_possible_grid()

    @abstractmethod
    def solve(self):
        pass

    def is_possible(self, field, number):

        row, col = self.get_field_pos(field)
        for i in range(0, self.size):
            if self.grid[row][i] == number:
                return False

        for i in range(0, self.size):
            if self.grid[i][col] == number:
                return False

        segment_start_row, segment_start_col = self.get_field_pos(field)

        for i in range(segment_start_row, segment_start_row + self.segment_size):
            for j in range(segment_start_col, segment_start_col + self.segment_size):
                if self.grid[i][j] == number:
                    return False

        return True

    def get_field_pos(self, field):
        return field // self.size, field % self.size

    def get_segment_pos(self, field):
        return (field // self.size - field // self.size) % self.segment_size, (field % self.size - field % self.size) % self.segment_size

    def get_possible(self, field):

        row, col = self.get_field_pos(field)
        segment_start_row, segment_start_col = self.get_field_pos(field)

        possible = np.arange(1, self.size + 1)
        return np.setdiff1d(possible, np.unique(np.array([
            self.grid[row, :], self.grid[:, col],
            self.grid[
            segment_start_row:segment_start_row+self.segment_size,
            segment_start_col:segment_start_col+self.segment_size
            ]
        ])))

    def get_possible_grid(self):
        self.possible_grid = np.empty([self.size], dtype=list)
        for i in self.size[0]:
            for j in self.size[1]:
                self.possible_grid[i, j] = list(self.get_field_pos(i*self.size[0]+j*self.size[1]))


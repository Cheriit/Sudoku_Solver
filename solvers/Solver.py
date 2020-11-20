from abc import abstractmethod

import numpy as np


class Solver:

    def __init__(self, grid):
        self.size = grid.shape[0]
        self.grid: np.ndarray = grid
        self.segment_size = np.floor(np.sqrt(self.size))

    @abstractmethod
    def solve(self):
        pass

    def get_field_pos(self, field):
        return field // self.size, field % self.size

    def get_segment_pos(self, row, col):
        return row - row % self.segment_size, col - col % self.segment_size

import numpy as np

from solvers.Solver import Solver


class Backtracking(Solver):

    def solve(self, field=0):
        row, col = self.get_field_pos(field)

        if field == self.size ** 2 - 1:
            return True

        if self.grid[row][col] > 0:
            self.solve(field + 1)

        for i in range(1, self.size + 1):
            if self.is_possible(field, i):
                self.grid[row][col] = i
                if self.solve(field + 1):
                    return True
        return False

    def is_possible(self, field, number):

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


class BacktrackingV2(Solver):

    def __init__(self, grid):
        super().__init__(grid)
        self.possible_grid = np.empty([self.size], dtype=list)
        self.fill_possible_grid()

    def solve(self, field=0):
        row, col = self.get_field_pos(field)

        if field == self.size ** 2 - 1:
            return True

        if self.grid[row][col] > 0:
            self.solve(field + 1)
        visited = []
        for value in self.possible_grid[row][col]:
            if not value in visited:
                self.grid[row][col] = value
                visited.push(value)
                if self.removePossible(row, col, value) and self.solve(field + 1):
                    return True
                else:
                    self.addPossible(row, col, value)
                
        return False

    def get_possible(self, field):

        row, col = self.get_field_pos(field)
        segment_start_row, segment_start_col = self.get_field_pos(field)

        possible = np.arange(1, self.size + 1)
        return np.setdiff1d(possible, np.unique(np.array([
            self.grid[row, :], self.grid[:, col],
            self.grid[
                segment_start_row: segment_start_row + self.segment_size,
                segment_start_col: segment_start_col + self.segment_size
            ]
        ])))

    def fill_possible_grid(self):
        for i in self.size:
            for j in self.size:
                self.possible_grid[i, j] = list(self.get_field_pos(i * self.size + j * self.size))

    def removePossible(self, row, col, value):
        res = True
        segment_start_row, segment_start_col = self.get_segment_pos(row, col)

        for i in self.possible_grid[row, col:]:
            i.remove(value)
            if not len(i):
                res = False

        for i in self.possible_grid[row:, col]:
            i.remove(value)
            if not len(i):
                res = False

        for i in np.nditer(self.possible_grid[
                           row: segment_start_row + self.segment_size,
                           col: segment_start_col + self.segment_size
                           ]):
                i.remove(value)
                if not len(i):
                    res = False
        return res

    def addPossible(self, row, col, value):
        segment_start_row, segment_start_col = self.get_segment_pos(row, col)

        for i in self.possible_grid[row, col:]:
            i.push(value)

        for i in self.possible_grid[row:, col]:
            i.push(value)

        for i in np.nditer(self.possible_grid[
                           row: segment_start_row + self.segment_size,
                           col: segment_start_col + self.segment_size
                           ]):
            i.push(value)


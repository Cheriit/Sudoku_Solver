import numpy as np

from solvers.solver import Solver


class Basic(Solver):

    def solve(self, field: int = 0) -> bool:
        row, col = self.get_field_pos(field)

        if field == self.size ** 2:
            return True

        if self.grid[row][col] > 0:
            return self.solve(field + 1)
        else:
            for i in range(1, self.size + 1):
                print("Processing ("+str(row)+", "+str(col)+"): "+str(i), end="\r")
                if self.is_possible(field, i):
                    self.grid[row][col] = i
                    if self.solve(field + 1):
                        return True
            self.grid[row][col] = 0
        return False


class Possible(Solver):

    def __init__(self, grid: np.ndarray):
        super().__init__(grid)
        self.possibility_grid: np.ndarray = np.empty([self.size, self.size], dtype=set)
        self.fill_possible_grid()

    def solve(self, field: int = 0) -> bool:
        row, col = self.get_field_pos(field)

        if field == self.size ** 2:
            return True

        if self.grid[row][col] > 0:
            return self.solve(field + 1)
        else:
            element_possibilities_set = self.possibility_grid[row][col]
            start_possibilities_set = element_possibilities_set.copy()
            while len(element_possibilities_set):
                possible_value = list(element_possibilities_set)[0]
                print("Processing ("+str(row)+", "+str(col)+"): "+str(possible_value), end="\r")
                self.grid[row][col] = possible_value
                if self.remove_possibility(row, col, possible_value):
                    if self.solve(field + 1):
                        return True
                    else:
                        self.grid[row][col] = 0
                        self.add_possibility_to_row(row, col, possible_value)
                        self.add_possibility_to_column(row, col, possible_value)
                        self.add_possibility_to_segment(row, col, possible_value)
                        element_possibilities_set.discard(possible_value)
        self.possibility_grid[row][col] = start_possibilities_set
        return False

    def get_possibility_set(self, row: int, col: int) -> set:

        segment_start_row, segment_start_col = self.get_segment_pos(row, col)
        segment = self.grid[
                  segment_start_row: segment_start_row + self.segment_size,
                  segment_start_col: segment_start_col + self.segment_size
                  ]

        possibilities_set = set(range(1, self.size + 1))
        possibilities_set = possibilities_set.difference(set(self.grid[row, :]))
        possibilities_set = possibilities_set.difference(set(self.grid[:, col]))
        possibilities_set = possibilities_set.difference({y for x in segment for y in x})
        possibilities_set.discard(0)
        return possibilities_set

    def fill_possible_grid(self) -> None:
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == 0:
                    self.possibility_grid[i, j]: set = self.get_possibility_set(i, j)

    def remove_possibility_from_row(self, row: int, col: int, value: int) -> bool:
        if col + 1 != self.size:
            for i in self.possibility_grid[row, col + 1:]:
                if i is not None:
                    i.discard(value)
                    if not len(i):
                        return False
        return True

    def remove_possibility_from_column(self, row: int, col: int, value: int) -> bool:
        if row + 1 != self.size:
            for i in self.possibility_grid[row + 1:, col]:
                if i is not None:
                    i.discard(value)
                    if not len(i):
                        return False
        return True

    def remove_possibility_from_segment(self, row: int, col: int, value: int) -> bool:
        segment_start_row, segment_start_col = self.get_segment_pos(row, col)
        current_field = self.get_field(row, col)

        for i in range(self.size):
            segment_row = segment_start_row + (i // self.segment_size)
            segment_col = segment_start_col + (i % self.segment_size)
            if self.possibility_grid[segment_row, segment_col] is not None and (current_field < self.get_field(row, col)):
                self.possibility_grid[segment_row, segment_col].discard(value)
                if not len(self.possibility_grid[segment_row, segment_col]):
                    return False
        return True

    def add_possibility_to_row(self, row: int, col: int, value: int) -> None:
        if col + 1 != self.size:
            for i in range(col + 1, self.size):
                if self.is_possible(self.get_field(row, i), value) \
                        and self.possibility_grid[row, i] is not None:
                    self.possibility_grid[row, i].add(value)

    def add_possibility_to_column(self, row: int, col: int, value: int) -> None:
        if row + 1 != self.size:
            for i in range(row, self.size):
                if self.is_possible(self.get_field(i, col), value) \
                        and self.possibility_grid[i, col] is not None:
                    self.possibility_grid[i, col].add(value)

    def add_possibility_to_segment(self, row: int, col: int, value: int) -> None:
        segment_start_row, segment_start_col = self.get_segment_pos(row, col)

        current_field = self.get_field(row, col)

        for i in range(self.size):
            segment_row = segment_start_row + (i // self.segment_size)
            segment_col = segment_start_col + (i % self.segment_size)
            if self.possibility_grid[segment_row, segment_col] is not None \
                    and current_field < self.get_field(segment_row, segment_col) \
                    and self.is_possible(self.get_field(segment_row, segment_col), value):
                self.possibility_grid[segment_row, segment_col].add(value)

    def remove_possibility(self, row: int, col: int, value: int) -> bool:
        element_possibilities_set = self.possibility_grid[row][col]

        if not self.remove_possibility_from_row(row, col, value):
            self.grid[row][col] = 0
            self.add_possibility_to_row(row, col, value)
            element_possibilities_set.discard(value)
            return False

        if not self.remove_possibility_from_column(row, col, value):
            self.grid[row][col] = 0
            self.add_possibility_to_row(row, col, value)
            self.add_possibility_to_column(row, col, value)
            element_possibilities_set.discard(value)
            return False

        if not self.remove_possibility_from_segment(row, col, value):
            self.grid[row][col] = 0
            self.add_possibility_to_row(row, col, value)
            self.add_possibility_to_column(row, col, value)
            self.add_possibility_to_segment(row, col, value)
            element_possibilities_set.discard(value)
            return False
        element_possibilities_set.discard(value)
        return True

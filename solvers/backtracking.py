from solvers.Solver import Solver


class Backtracking(Solver):

    def solve(self, field=0):
        row, col = self.get_field_pos(field)

        if field == self.size[0] ** 2 - 1:
            return True

        if self.grid[row][col] > 0:
            self.solve(field + 1)

        for i in range(1, self.size[0] + 1):
            if self.is_possible(self.grid, field, i):
                self.grid[row][col] = i;
                if self.solve(field + 1):
                    return True
        return False

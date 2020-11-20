from solvers.helpers import is_possible


def solve(grid, field=0):
    size = len(grid)
    row = field // size
    col = field % size

    if field == size ** 2 - 1:
        return True

    if grid[row][col] > 0:
        solve(grid, field + 1)

    for i in range(1, size + 1):
        if is_possible(grid, field, i):
            grid[row][col] = i;
            if solve(grid, field + 1):
                return True

    return False

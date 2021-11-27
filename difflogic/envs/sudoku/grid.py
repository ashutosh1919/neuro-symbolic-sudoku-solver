"""Implement random grid generators and Grid class."""

import copy
import numpy as np
import pandas as pd

import jacinle.random as random

__all__ = ['Grid', 'randomly_generate_grid_from_data', 'get_random_grid_generator']

dataset = pd.read_csv('difflogic/envs/sudoku/sudoku.csv', dtype={0:'str', 1:'str'})

class Grid:
  """
  Store n x n Grid as numpy matrix
  """

  def __init__(self, dim, nr_empty, grid):
    self._dim = dim
    self._nr_empty = nr_empty
    self._grid = grid
    self._solved_grid = None
    self._optimal_steps = None

  @property
  def dim(self):
    return self._dim

  @property
  def nr_empty(self):
    return self._nr_empty

  @property
  def optimal_steps(self):
    return self._optimal_steps
    
  def get_grid(self):
    return copy.copy(self._grid)
    
  def get_empty_coordinates(self, grid):
    return np.argwhere(grid == 0)
    
  def is_row_valid(self, grid, row, num):
    assert row < self.dim, 'row must be inside the grid'
    return len(np.argwhere(grid[row] == num)) == 1
    
  def is_column_valid(self, grid, col, num):
    assert col < self.dim, 'column must be inside the grid'
    return len(np.argwhere(grid[:, col] == num)) == 1
    
  def is_submat_valid(self, grid, row, col, num):
    assert row < self.dim and col < self.dim, 'sub matrix must be inside the grid'
    return len(np.argwhere(grid[row:(row+3), col:(col+3)] == num)) == 1
    
  def recursively_solve(self, grid, empty_coords, index):
    if index == len(empty_coords):
      return True

    row, col = empty_coords[index]
    for num in range(1, self.dim + 1):
      grid[row, col] = num
      if self.is_row_valid(grid, row, num) and self.is_column_valid(grid, col, num)  and self.is_submat_valid(grid, (row//3)*3, (col//3)*3, num):
        self._optimal_steps += 1
        if self.recursively_solve(grid, empty_coords, index + 1):
          return True
      grid[row, col] = 0
        
    return False
    
  def get_solved_grid(self):
    if self._solved_grid is not None:
      return self._solved_grid
    self._optimal_steps = 0
    grid = self.get_grid()
    empty_coords = self.get_empty_coordinates(grid)
    self.recursively_solve(grid, empty_coords, 0)
    self._solved_grid = copy.copy(grid)
    return self._solved_grid


def randomly_generate_grid_from_data(nr_empty, dim=9):
  """
  Randomly generate grid by sampling data and making some of the
  cells empty intentionally.
  """
  sample_index = random.randint(0, len(dataset))
  grid_str = dataset.loc[sample_index, 'solutions']
  grid = np.array([int(x) for x in grid_str])
  grid = grid.reshape((9, 9))
  
  for _ in range(nr_empty):
    row, col = 0, 0
    while True:
      index = random.randint(0, 81)
      row, col = index // dim, index % dim
      if grid[row, col] != 0:
        break
    grid[row, col] = 0
  
  return Grid(dim, nr_empty, grid)


def get_random_grid_generator():
  return randomly_generate_grid_from_data


if __name__ == '__main__':
  grid = randomly_generate_grid_from_data(15)
  print(grid.get_grid())
  print(grid.get_solved_grid())
  print(grid.optimal_steps)
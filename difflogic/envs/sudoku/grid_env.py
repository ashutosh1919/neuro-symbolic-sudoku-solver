"""The environment class for grid tasks."""

import copy
import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase
from .grid import get_random_grid_generator

__all__ = ['GridEnvBase', 'SudokuGridEnv']


class GridEnvBase(SimpleRLEnvBase):
  """The base class for Grid Environment."""

  def __init__(self,
               nr_empty,
               dim=9):

    super().__init__()
    self._nr_empty = nr_empty
    self.dim = dim
    self._grid = None   

  @notnone_property
  def graph(self):
    return self._graph

  @property
  def nr_empty(self):
    return self._nr_empty

  def _restart(self):
    """Restart the environment."""
    self._gen_grid()

  def _gen_grid(self):
    """generate the grid by specified method."""
    gen = get_random_grid_generator()
    self._grid = gen(self.nr_empty, self.dim)


class SudokuGridEnv(GridEnvBase):
  """Env for Finding a path from starting node to the destination."""
  def __init__(self,
               nr_empty,
               dim=9):
    super().__init__(nr_empty, dim)
    self._empty_cells = None
  
  @property
  def empty_cells(self):
    return self._empty_cells
  
  @property
  def optimal_steps(self):
    return self._optimal_steps
  
  @property
  def solved(self):
    return self._solved

  def _restart(self):
    super()._restart()
    self._solved = self._grid.get_solved_grid()
    self._empty_cells = self._grid.get_empty_coordinates(self._grid.get_grid())
    self._optimal_steps = self._grid.optimal_steps
    self._current = self._grid.get_grid()
    self._set_current_state(self._grid.get_grid())
    self._steps = 0
  
  def _gen(self):
    grid = self._grid.get_grid()
    st, ed = np.where(grid == self._empty_cell)
    if len(st) == 0:
      return None
    ind = random.randint(len(st))
    return st[ind], ed[ind], 1

  def _action(self, action):
    if np.array_equal(self._current, self._solved):
      return 1, True
    print("Action", action)
    empty_cell_index, num = action
    # print(self.nr_empty, self._empty_cells, self.current_state)
    row, col = self._empty_cells[empty_cell_index]
    print(row, col)
    print(self.current_state)
    # row, col, num = action
    grid = copy.copy(self._current)
    # if grid[row, col] == 0:
    grid[row, col] = num
    if self._grid.is_row_valid(grid, row, num) and self._grid.is_column_valid(grid, col, num) and self._grid.is_submat_valid(grid, (row//3)*3, (col//3)*3, num):
      self._current = grid
    self._set_current_state(self._current)
    print(self.current_state)
    if np.array_equal(self._current, self._solved):
      return 1, True
    self._steps += 1
    # if self._steps >= self._optimal_steps:
    #   return 0, True
    return 0, False
    

    
    

    




      

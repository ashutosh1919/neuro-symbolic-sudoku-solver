"""The environment class for grid tasks."""

import numpy as np

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase

__all__ = ['GridEnvBase', 'SudokuGridEnv']


class GridEnvBase(SimpleRLEnvBase):
  """The base class for Grid Environment."""

  def __init__(self,
               nr_empty):
    super().__init__()
    self._nr_empty = nr_empty
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
    n = self._nr_empty
    gen = get_random_grid_generator()
    self._grid = gen(n)
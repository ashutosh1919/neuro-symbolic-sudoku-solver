"""Quick access for graph environments."""

from jaclearn.rl.proxy import LimitLengthProxy

from .grid_env import SudokuGridEnv
from ..utils import get_action_mapping_sudoku
from ..utils import MapActionProxy

__all__ = ['get_grid_env', 'make']


def get_grid_env(nr_empty, dim=9):
  env_cls = SudokuGridEnv
  p = env_cls(nr_empty, dim)
  p = LimitLengthProxy(p, 400)
  mapping = get_action_mapping_sudoku(nr_empty, dim, exclude_self=False)
  # print(mapping)
  p = MapActionProxy(p, mapping)
  return p

def make(task, *args, **kwargs):
  if task == 'sudoku':
    return get_grid_env(*args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))
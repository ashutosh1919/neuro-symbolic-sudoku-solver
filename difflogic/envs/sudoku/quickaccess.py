"""Quick access for graph environments."""

from .grid_env import SudokuGridEnv

__all__ = ['get_grid_env', 'make']


def get_grid_env(nr_empty, dim=9):
  env_cls = SudokuGridEnv
  p = env_cls(
      nr_empty, dim
  )
  return p


def make(task, *args, **kwargs):
  if task == 'sudoku':
    return get_grid_env(*args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))
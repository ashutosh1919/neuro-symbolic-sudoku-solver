import time
import copy
import numpy as np
import pandas as pd

import jacinle.random as random
from jacinle.utils.meta import notnone_property
from jaclearn.rl.env import SimpleRLEnvBase
from .grid import get_random_grid_generator
from .grid import get_solved_grid

__all__ = ['Grid', 'randomly_generate_grid_from_data', 'get_random_grid_generator']

dataset = pd.read_csv('difflogic/envs/sudoku/sudoku.csv', dtype={0:'str', 1:'str'})

class BacktrackGrid(Grid):
    def __init__(self,nr_runs) -> None:
        super().__init__()
        self.nr_runs = nr_runs
        self.arr = []
        self._time_backtrack = None

    
    def perform_backtrack():
        for i in range(self.nr_runs):
            # a not required to be global
            a, self._time_backtrack = get_solved_grid()
            self.arr.append(self._time_backtrack)
        return self.arr


            

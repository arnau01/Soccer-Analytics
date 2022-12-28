import os   
import random
import warnings

import numpy as np
import pandas as pd


class OptaPipeline:
    """Class for building the StatsBomb pipeline"""

    def __init__(self,use_atomic = False, data_dir = "../pkl_data/Opta/ws_ALL_17-22"):
        self.use_atomic = use_atomic
        self.data_dir = data_dir
        if use_atomic:
            self.data_dir = data_dir+"_atom.pkl"
        else:
            self.data_dir = data_dir+"_spadl.pkl"

    def run_pipeline(self):
            
        df = pd.read_pickle(self.data_dir)
        return df
# ! pip install mplsoccer

import os
import random
import warnings

import matplotlib.pyplot as plt
import matplotsoccer as mps
import numpy as np
import pandas as pd
import scipy
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
import seaborn as sns
from socceraction.data.statsbomb import StatsBombLoader
from tqdm import tqdm
tqdm.pandas()
import bin_action_seq as bs
import gen_plot as gp

# Find similar sequences
# Generate a heatmap for the set of similar actions
# Then find what happens in all the next actions i



warnings.filterwarnings('ignore')
# Amount of bins to adjust heatmap
X_B = bs.X_B
Y_B = bs.Y_B
M = bs.M
REBUILD_DATA = bs.REBUILD_DATA
# Amount of start actions to check similarity
ST = 3
bin_data = bs.file_name


def load_data():
    
    action_data = np.load(bin_data, allow_pickle=True)
    seq_names = action_data.files
    # print(len(seq_names))
    for name in seq_names:
        # print(name)
        np_data = action_data[name]
        # print(np_data)

    b = np.delete(np_data, [0,2], axis=2)
    b = b.reshape((b.shape[0], b.shape[1] * b.shape[2]))
    df_b = pd.DataFrame(b)

    # Get (x,y) tuple for heatmap
    x_y = np.delete(np_data, [0,1], axis=2)
    x_y = x_y.reshape((x_y.shape[0], x_y.shape[1] * x_y.shape[2]))
    df_xy = pd.DataFrame(x_y)

    return df_b, df_xy


# Finds actions which are exactly the same for the first ST actions

def find_sim(df_b):

    df_start = df_b.iloc[:, :ST]
    # Get matching indices of duplicate rows
    # df_start = df_start[df_start.duplicated(keep=False)]
    matches = df_start.groupby(list(df_start)).apply(lambda x: tuple(x.index)).tolist()
   
    # Sort list by length
    matches.sort(key=len, reverse=True)

    return matches

# Find similar actions by rounding to lowest and upper 5
# Checks if two rows are similar have the same firs ST actions

def find_similar_rows_round(df):
    round_up_to_5 = lambda x: x + 5 - x % 5

    # define a lambda function that rounds the value down to the nearest 5
    round_down_to_5 = lambda x: x - x % 5

    # create a new dataframe with rounded values
    # df_rounded = df.progress_apply(lambda x: (round_down_to_5(x), round_up_to_5(x)))

    df_rounded = df.apply(lambda x: round_down_to_5(x))

    return find_sim(df_rounded) 
 

if __name__ == '__main__':

    # If file name exists, load it
    if os.path.exists(bin_data) and not REBUILD_DATA:
        df_b,df_xy = load_data()
    
    # If bin file doesn't exist, create it
    else:
        # Call main method from bin_action_seq.py
        bs.main()
        df_b,df_xy = load_data()
    
    print("Finding exact same start sequences")
    # Add conditions like attacking half
    matches = find_sim(df_b)
    
    print("Finding similar sequences with rounding of bins")
    sim_round = find_similar_rows_round(df_b)

    if len(matches) == 0:
        print("No similar sequences")
    # Generate heatmap for similar sequences

    for i in range(len(matches[:3])) :        
        gp.generate_hm(matches[i],df_xy)
        if len(matches[i]) > 5:
            gp.generate_kde(matches[i],df_xy)
    
    gp.generate_charts(matches)
    gp.generate_charts(sim_round)
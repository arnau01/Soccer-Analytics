# ! pip install mplsoccer
#%%
import os
import random
import warnings

import itertools
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
import bin_action_seq as bs
import gen_plot as gp

# Find similar sequences
# Generate a heatmap for the set of similar actions
# Then find what happens in all the next actions i


#%%
warnings.filterwarnings('ignore')
# Amount of bins to adjust heatmap
X_B = bs.X_B
Y_B = bs.Y_B
M = bs.M
REBUILD_DATA = bs.REBUILD_DATA
# Amount of start actions to check similarity
ST = 2
bin_data = bs.file_name



def load_data():
    print("Loading bin file :", bin_data)
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



def unique_seq(matches):
    # Get unique sequences (I.e sequences which have 1 match)
    unique = []
    # Order matches in ascending order
    matches.sort(key=len, reverse=True)
    for i in range(len(matches)):
        if len(matches[i])==1:
            unique.append(matches[i])

    
    return unique


def check_half(df):
    # df['off'] = df.apply(lambda row: row[0][0] > 4 and row[1][0] > 4, axis=1)
    # Print percentage of sequences where both actions are in the attacking half 2dp
    df['off'] = (df.iloc[:, :ST].applymap(lambda x: x[0]) > 4).all(axis=1)
    # Only print two decimal places
    print("Percentage of sequences where first actions are in the attacking half: ", 
    round(df['off'].sum()/len(df)*100,2), "%")
    
    return df

    
    

#%%

if __name__ == '__main__':
    
    # If file name exists, load it
    
    if os.path.exists(bin_data) and not REBUILD_DATA:
        df_b,df_xy = load_data()
        print("Number of sequences: ", len(df_b))
        # Non-duplicate sequences
        print("Number of non-duplicate sequences: ", len(df_b.drop_duplicates()))
        df_s = df_b.iloc[:, :ST]
        # Only take into account last three columns (start actions)
        non_dup = df_s.drop_duplicates()
        print("Number of non-duplicate start sequences: ", len(non_dup))
        
    
    # If bin file doesn't exist, create it
    else:
        
        # Call main method from bin_action_seq.py
        bs.main()
        df_b,df_xy = load_data()
    
    print("Finding exact same start sequences")
    # Add conditions like attacking half
    matches = find_sim(df_b)
    
    # Print smallest and largest bin in each column
    # print(df_b.min())
    # print(df_b.max())

    if len(matches) == 0:
        print("No similar sequences")
    # Generate heatmap for similar sequences

    for i in range(len(matches[:3])) :        
        gp.generate_hm(matches[i],df_xy)

        # Generate kde for sequences with more than 5 matches
        # if len(matches[i]) > 5:
        #     gp.generate_kde(matches[i],df_xy)
    
    gp.generate_charts(matches)
    # gp.generate_charts(sim_round)
    # %%
    unique = unique_seq(matches)
    print("Unique sequences: ", len(unique))
    # Flatten all unique sequences to a single array
    unique = list(itertools.chain.from_iterable(unique))


    # Filter the df_s dataframe to only include unique sequences (indexes in unique)
    df_xy_u = df_xy.iloc[unique]

    

    df_xy_u = df_xy_u.iloc[:, :ST]
    

    # df_off = check_half(df_xy_u)

    # Amount of sequences with less than 10 matches
    # Sort matches in ascending order
    matches.sort(key=len, reverse=False)
    counter = 0
    for i in range(len(matches)):
        if len(matches[i]) < 10:
            counter += 1
        else:
            break
    print("Number of sequences with less than 10 matches: ", counter)

    #%%

    # Generate a KDE of where unique sequences happen
    gp.generate_kde_unique(df_xy_u)
    
# %%

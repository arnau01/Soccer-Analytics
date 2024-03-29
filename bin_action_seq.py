# Uncomment to install required modules

# ! pip install mplsoccer --user
# ! pip install socceraction --user
# ! pip install statsbombpy

#%%
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from tqdm import tqdm

import sb_pipeline
import wy_pipeline
import opta_pipeline
#%%
USE_ATOMIC = False
DOWNLOAD_RAW_DATA = False
REBUILD_DATA = False
OFFENSIVE = False
X_B = 8
Y_B = 6
M = 3

if not os.path.isdir('./seq'):
        os.makedirs('./seq')
file_name = './seq/b_u_data_seq_'+str(X_B)+'_'+str(M)+str(USE_ATOMIC)+str(OFFENSIVE)+'.npz'
warnings.filterwarnings('ignore')
# Using API to get data from StatsBomb
SBL = StatsBombLoader()



# Create a list of n sequences
# (n being a variable which is set to say 1000).
# Where each sequence is created by picking a random time
# Pick random time , random period , random game
# Extracts the next to m actions (m being a variable,
# say 10). Each sequence is saved as a numpy array with 3 columns
# time (from first action), x, y position of the ball
# at each subsequent action. 
# Save to single npz file

def rolling_window(a, window, slide):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::slide]

def seq_array(df,n=150000000, m=10):

    
    # sort by game id and time
    df = df.sort_values(by=["game_id", "period_id", "time_seconds"])
    
    if USE_ATOMIC:
        simple_actions_allowed = ["pass", "cross", "dribble", "shot", "take_on", "receival", "goal"]
        x_var_name = 'x'
        y_var_name = 'y'
    else:
        simple_actions_allowed = ["pass", "cross", "dribble", "shot"]
        x_var_name = 'start_x'
        y_var_name = 'start_y'

    # print(df.columns)

    df[x_var_name] = (pd.cut(df[x_var_name], bins=X_B, labels=False)+1) #* (105/x_b)
    df[y_var_name] = pd.cut(df[y_var_name], bins=Y_B, labels=False)+1 #* (105/x_b)
    # Create a tuple x and y
    df["xy"] = list(zip(df[x_var_name], df[y_var_name]))
    # Create new column for unique bin id
    df["bin_id"] = df[x_var_name] + (df[y_var_name] * X_B)
    # I feel like there is a way of doing this without a for loop !
    sequences = []

    
    for game_id, df_game in tqdm(df.groupby("game_id")):

        if USE_ATOMIC:
            # filter out rows where dx is zero and dy is zero
            df_game = df_game.loc[(df_game["dx"] != 0) | (df_game["dy"] != 0)]

        # get all the time, x, y positions
        # df_actions = df_game[["time_seconds", "type_name", "team_id", x_var_name, y_var_name]]
        df_actions = df_game[["time_seconds", "type_name", "team_id", "bin_id", x_var_name, y_var_name, "xy"]]

        # create the sliding windows
        a = rolling_window(np.arange(len(df_actions)), m, (np.rint(m / 2)).astype(int)-1)

        # Probably a better way to do this
        df_test = pd.concat([pd.DataFrame(df_actions[x].to_numpy()[a]).rename(columns=lambda y: f'{x}_{y + 1}')
                         for x in df_actions.columns], axis=1)


        # get all column names that include "team_name_"
        team_col_names = [col for col in df_test.columns if "team_id" in col]
        action_col_names = [col for col in df_test.columns if "type_name" in col]
        x_col_names = [col for col in df_test.columns if x_var_name in col]

        # Ensure only permitted actions, same team keep the ball and only in the attacking half
        df_test_filtered = df_test.loc[df_test[action_col_names].isin(simple_actions_allowed).all(axis=1) 
                                        & df_test[team_col_names].nunique(axis=1) == 1]
                                        
        if OFFENSIVE:
            df_test_filtered = df_test_filtered[df_test_filtered[x_col_names].gt(int(X_B/2)).all(axis=1)]                               

        # extract just the time, x, y columns
        time_second_names = [col for col in df_test_filtered.columns if "time_seconds_" in col]

        x_names = [col for col in df_test_filtered.columns if x_var_name+"_" in col]
        y_names = [col for col in df_test_filtered.columns if y_var_name+"_" in col]
        bin_id_names = [col for col in df_test_filtered.columns if "bin_id_" in col]
        xy_names = [col for col in df_test_filtered.columns if "xy_" in col]
        # team_ids = [col for col in df_test_filtered.columns if "team_id_" in col]

        df_test_filtered = df_test_filtered[time_second_names + bin_id_names + xy_names ]

        

        # convert every row of a dataframe to a numpy matrix of a given size - THIS IS DUMB WAY OF DOING IT
        for index, row in df_test_filtered.iterrows():


            action_matrix = row.to_numpy().reshape(3, m).transpose()
            # Condition to ensure that the sequence is only of same team
            # Check all the items in 4th column are the same
            # if len(np.unique(action_matrix[:, 3])) != 1:
            #     # If they are not the same, then the ball has passed to the opposition
            #     continue
            # Drop 4th column which is team name
            action_matrix = action_matrix[:, :3]
            # TO DO - DO THIS OUTSIDE LOOP
            time = action_matrix[0,0]
            # subtract the time from the first action from all the actions
            action_matrix[:,0] -= time

            # FUDGE TO ENSURE THAT YOU DON'T GET IDENTICAL TIMES
            vpairs = (action_matrix[:-1, 0] == action_matrix[1:, 0])# append to sequences
            vpairs = np.append(False, vpairs)
            # for a masked array, add random values to the masked values
            # this is to ensure that the time is not zero
            action_matrix[:, 0] = np.ma.masked_array(action_matrix[:, 0], mask=vpairs).filled(action_matrix[:, 0] + np.random.uniform(0.0, 0.5))

            sequences.append(action_matrix)
            
    # shuffle the sequences
    random.shuffle(sequences)

    if n < len(sequences):
       sequences = sequences[:n]
    
    # save to npz file
    np.savez_compressed(file_name, sequences)

#%%
def main():
    random.seed(42)
    
    if not os.path.isfile(file_name) or REBUILD_DATA:
        sb_df = pd.DataFrame()
        wy_df = pd.DataFrame()
        # Importing SB data
        print("Importing SB data")
        sb = sb_pipeline.SBPipeline(use_atomic = USE_ATOMIC)
        sb_df = sb.run_pipeline()

        # Importing Wyscout data
        # wy = wy_pipeline.wyPipeline(use_atomic = USE_ATOMIC)
        # print("Importing Wyscout data")
        # wy_df = wy.run_pipeline()
        
        # Importing Opta data
        # print("Importing Opta data")
        # op = opta_pipeline.OptaPipeline()
        # op_df = op.run_pipeline()

        df = sb_df

        # Concatenate the two dataframes
        # df = pd.concat([sb_df, op_df])

        # group the DataFrame by game_id, and count the number of events in each group
        game_event_counts = df.groupby("game_id").size().reset_index(name='event_count')

        # filter out games with less than 40 events
        complete_games = game_event_counts[game_event_counts['event_count'] >= 40]

        # use the game_id's from the `complete_games` DataFrame to filter the original DataFrame
        df = df[df['game_id'].isin(complete_games['game_id'])]
        
        # Total events in the dataset : 
        print("Total events in the dataset : ", len(df))
        
        random.seed(42)
        print("Creating sequences...")
        seq_array(df, n=150000, m=M)
    else:
        print(f"Loading existing {file_name} file!")
        action_data = np.load(file_name,allow_pickle=True)
        seq_names = action_data.files
        print(len(seq_names))
        for name in seq_names:
            print(name)
            np_data = action_data[name]
            print(np_data)

#%%
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()
# %%

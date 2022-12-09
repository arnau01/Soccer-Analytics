# Uncomment to install required modules

# ! pip install mplsoccer --user
# ! pip install socceraction --user
# ! pip install statsbombpy

import os
import random
import warnings

import numpy as np
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from tqdm import tqdm

USE_ATOMIC = False
DOWNLOAD_RAW_DATA = False
REBUILD_DATA = False
X_B = 16
Y_B = 12
M = 4

if not os.path.isdir('./seq'):
        os.makedirs('./seq')
file_name = './seq/b_u_data_seq_'+str(X_B)+'_'+str(M)+str(USE_ATOMIC)+'.npz'
warnings.filterwarnings('ignore')
# Using API to get data from StatsBomb
SBL = StatsBombLoader()
# Get all the competion and season id's from the Barcelona games
comps = SBL.competitions()
comps = comps.loc[comps["competition_name"] == "La Liga"]
comp_season_id = list(zip(comps.competition_id, comps.season_id))


# Function to get all the game id's for a given competition and season
def get_game_ids(comp_season_id):
    all_games = []
    for i in range(len(comp_season_id)):
        comp_id = comp_season_id[i][0]
        season_id = comp_season_id[i][1]
        games = SBL.games(comp_id, season_id)
        all_games += list(games.game_id)
    return all_games



# Function to get all actions for a given set of game id's
def get_all_actions(games):
    all_actions = []
    for g in tqdm(games, total = len(games)):
        events = SBL.events(g)
        players = SBL.players(g)
        players = players[['player_id', 'player_name', 'nickname']].drop_duplicates(subset='player_id')
        teams = SBL.teams(g)
        df_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=777)
        atomic_actions = atomicspadl.convert_to_atomic(df_actions)
        atomic_actions = (
        atomic_actions
        .merge(atomicspadl.actiontypes_df(), how="left")
        .merge(spadl.bodyparts_df(), how="left")
        .merge(players, how="left")
        .merge(teams, how="left")
        )
        atomic_actions["player_name"] = atomic_actions[["nickname", "player_name"]].apply(lambda x: x[0] if x[0] else x[1],axis=1)
        del atomic_actions['nickname']
        all_actions.append(atomic_actions)
    df = pd.concat(all_actions)
    return df


# Create a list of n sequences
# (n being a variable which is set to say 1000).
# Where each sequence is created by picking a random time
# Pick random time , random period , random game
# Extracts the next to m actions (m being a variable,
# say 10). Each sequence is saved as a numpy array with 3 columns
# time (from first action), x, y position of the ball
# at each subsequent action. In terms of matches lets extract them all
# from the Barcelona games for now.
# Save to single npz file

def rolling_window(a, window, slide):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::slide]

def seq_array(n=1000, m=10):

    if USE_ATOMIC:
        simple_actions_allowed = ["pass", "cross", "dribble", "shot", "take_on", "receival", "goal"]
        x_var_name = 'x'
        y_var_name = 'y'
    else:
        simple_actions_allowed = ["pass", "cross", "dribble", "shot"]
        x_var_name = 'start_x'
        y_var_name = 'start_y'

    # check if file exists
    # Change this for to apply for all .pkl files
    if not os.path.isfile('barca_data_atomic.pkl') or DOWNLOAD_RAW_DATA:

        print("Downloading raw data from StatsBomb")
        all_games = get_game_ids(comp_season_id)

        # Create a df of all games
        # This is quicker than loading single random games afterwards
        df = get_all_actions(all_games)

        # Save to pickle
        df.to_pickle('pkl/barca_data_atomic.pkl')

    else:
        # df = pd.read_pickle("pkl/barca_data_atomic.pkl")
        print("Loading data from pickle")
        sb_df = pd.read_pickle("../pkl_data/sb_all_data.pkl")
        wy_df = pd.read_pickle("../pkl_data/wy_all_data.pkl")
        df = pd.concat([sb_df, wy_df])

    # # filter for Barcelona games
    # Removed condition for now to check if Barcelona keep the ball
    # df = df.loc[df["team_name"] == "Barcelona"]

    # sort by game id and time
    df = df.sort_values(by=["game_id", "period_id", "time_seconds"])

    # Create a new column for the binned x and binned y
    # Apply a scale to the binned columns
    # Added 1 to so values can finish at 105 (end of pitch) 
    # TO DO: condition for goals to be in correct bin
    x_b = X_B
    y_b = Y_B
    df["x"] = (pd.cut(df[x_var_name], bins=x_b, labels=False)+ 1) #* (105/x_b)
    df["y"] = pd.cut(df[y_var_name], bins=y_b, labels=False) #* (105/x_b)
    # Create a tuple x and y
    df["xy"] = list(zip(df["x"], df["y"]))
    # Create new column for unique bin id
    df["bin_id"] = df["x"] + (df["y"] * x_b)
    
    
    team_allowed = ['Barcelona']
    simple_actions_allowed = ["pass", "cross", "dribble", "shot", "take_on", "receival", "goal"]

    # I feel like there is a way of doing this without a for loop !
    sequences = []
    for game_id, df_game in tqdm(df.groupby("game_id")):

        # get all the time, x, y positions
        # df_actions = df_game[["time_seconds", "type_name", "team_name", "x", "y"]]
        df_actions = df_game[["time_seconds", "type_name", "team_id", "bin_id", "x", "y", "xy"]]
        # create the sliding windows
        a = rolling_window(np.arange(len(df_actions)), m, (np.rint(m / 2)).astype(int))

        # Probably a better way to do this
        df_test = pd.concat([pd.DataFrame(df_actions[x].to_numpy()[a]).rename(columns=lambda y: f'{x}_{y + 1}')
                         for x in df_actions.columns], axis=1)


        # get all column names that include "team_name_"
        team_col_names = [col for col in df_test.columns if "team_id" in col]
        action_col_names = [col for col in df_test.columns if "type_name" in col]

        # Ensure only permitted actions and Barcelona keep the ball are included (and not the opposition) | Want to change this to be same team keeps the ball
        # df_test_filtered = df_test.loc[df_test[action_col_names].isin(simple_actions_allowed).all(axis=1) & df_test[team_col_names].isin(team_allowed).all(axis=1)]
        df_test_filtered = df_test.loc[df_test[action_col_names].isin(simple_actions_allowed).all(axis=1) & df_test[team_col_names].nunique(axis=1) == 1]
        # extract just the time, x, y columns
        time_second_names = [col for col in df_test_filtered.columns if "time_seconds_" in col]
        x_names = [col for col in df_test_filtered.columns if "x_" in col]
        y_names = [col for col in df_test_filtered.columns if "y_" in col]
        bin_id_names = [col for col in df_test_filtered.columns if "bin_id_" in col]
        # Create a tuple of x and y positions
        xy_names = [col for col in df_test_filtered.columns if "xy_" in col]
        df_test_filtered = df_test_filtered[time_second_names + bin_id_names + xy_names]
        
        # convert every row of a dataframe to a numpy matrix of a given size - THIS IS DUMB WAY OF DOING IT
        for index, row in df_test_filtered.iterrows():
            action_matrix = row.to_numpy().reshape(3, m).transpose()

            # TO DO - DO THIS OUTSIDE LOOP
            time = action_matrix[0,0]
            # subtract the time from the first action from all the actions
            action_matrix[:, 0] = action_matrix[:, 0] - time

            sequences.append(action_matrix)


    # shuffle the sequences
    random.shuffle(sequences)

    if n < len(sequences):
        sequences = sequences[:n]

    # save to npz file
    np.savez_compressed(file_name, sequences)

def main():
    random.seed(42)

    # if file exists then load it
    if not os.path.isfile(file_name) or REBUILD_DATA:
        print("Creating data...")
        seq_array(n=200000000, m=M)
    else:
        action_data = np.load(file_name)
        seq_names = action_data.files
        print(len(seq_names))
        for name in seq_names:
            print(name)
            np_data = action_data[name]
            print(np_data)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    main()
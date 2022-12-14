import os
import random
import warnings

import numpy as np
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
from socceraction.data.wyscout import PublicWyscoutLoader
from tqdm import tqdm

import sb_pipeline as sb

warnings.filterwarnings('ignore')

DOWNLOAD_RAW_DATA = False
# USE_ATOMIC = sb.USE_ATOMIC
# Could add import all data here condition

class wyPipeline:
    """Class for building the Wyscout pipeline"""

    def __init__(self,use_atomic = False):
        self.W = PublicWyscoutLoader()
        self.use_atomic = use_atomic
    
    # Function to get all the competitions and seasons for wy
    def wy_get_all_comp_seasons(self):
        comps = self.W.competitions()
        comp_season_id = list(zip(comps.competition_id, comps.season_id))
        return comp_season_id

    # Function to get all the game id's for a given competition and season
    def get_game_ids(self,comp_season_id):
        all_games = []
        for i in range(len(comp_season_id)):
            comp_id = comp_season_id[i][0]
            season_id = comp_season_id[i][1]
            games = self.W.games(comp_id, season_id)
            all_games += list(games.game_id)
        return all_games
    
    
    
    # Function to get all actions for a given set of game id's
    def get_all_atomic_actions(self,games):
        all_actions = []
        for g in tqdm(games, total = len(games)):
            events = self.W.events(g)
            players = self.W.players(g)
            players = players[['player_id', 'player_name']].drop_duplicates(subset='player_id')
            teams = self.W.teams(g)
            df_actions = spadl.wyscout.convert_to_actions(events, home_team_id=777)
            atomic_actions = atomicspadl.convert_to_atomic(df_actions)
            atomic_actions = (
            atomic_actions
            .merge(atomicspadl.actiontypes_df(), how="left")
            .merge(spadl.bodyparts_df(), how="left")
            .merge(players, how="left")
            .merge(teams, how="left")
            )
            all_actions.append(atomic_actions)
        df = pd.concat(all_actions)
        return df
    
    
    def get_all_actions(self,games):
        all_actions = []

        for g in tqdm(games, total = len(games)):
            events = self.W.events(g)
            players = self.W.players(g)
            players = players[['player_id', 'player_name']].drop_duplicates(subset='player_id')
            teams = self.W.teams(g)
            df_actions = spadl.wyscout.convert_to_actions(events, home_team_id=777)
            actions = (
            df_actions
            .merge(spadl.actiontypes_df(), how="left")
            .merge(spadl.bodyparts_df(), how="left")
            .merge(players, how="left")
            .merge(teams, how="left")
            )
            all_actions.append(df_actions)
        df = pd.concat(all_actions)
        return df
    
    def run_pipeline(self):
        comp_season_id = self.wy_get_all_comp_seasons()
        games = self.get_game_ids(comp_season_id)
        if self.use_atomic :
            # check if file exists
            if not os.path.isfile('../pkl_data/wy_all_data_atomic.pkl') or DOWNLOAD_RAW_DATA:
                df = self.get_all_atomic_actions(games)
                df.to_pickle('../pkl_data/wy_all_data_atomic.pkl')
            else:
                df = pd.read_pickle("../pkl_data/wy_all_data_atomic.pkl")
            
        else:
            # check if file exists
            if not os.path.isfile('../pkl_data/wy_all_data.pkl') or DOWNLOAD_RAW_DATA:
                df = self.get_all_actions(games)
                df.to_pickle('../pkl_data/wy_all_data.pkl')
            else:
                df = pd.read_pickle("../pkl_data/wy_all_data.pkl")
                df = spadl.add_names(df)
        return df
import os
import random
import warnings

import numpy as np
import pandas as pd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from tqdm import tqdm

warnings.filterwarnings('ignore')

DOWNLOAD_RAW_DATA = False
USE_ATOMIC = True
IMPORT_ALL_DATA = True

class SBPipeline:
    """Class for building the StatsBomb pipeline"""

    def __init__(self):
        self.SBL = StatsBombLoader()
    
    # Function to get all the competitions and seasons for SB
    def sb_get_all_comp_seasons(self):
        comps = self.SBL.competitions()
        # In case we only want barcelona games
        if not IMPORT_ALL_DATA:
            comps = comps.loc[comps["competition_name"] == "La Liga"]
        comp_season_id = list(zip(comps.competition_id, comps.season_id))
        return comp_season_id

    # Function to get all the game id's for a given competition and season
    def get_game_ids(self,comp_season_id):
        all_games = []
        for i in range(len(comp_season_id)):
            comp_id = comp_season_id[i][0]
            season_id = comp_season_id[i][1]
            games = self.SBL.games(comp_id, season_id)
            all_games += list(games.game_id)
        return all_games
    
    
    
    # Function to get all actions for a given set of game id's
    def get_all_atomic_actions(self,games):
        all_actions = []
        for g in tqdm(games, total = len(games)):
            events = self.SBL.events(g)
            players = self.SBL.players(g)
            players = players[['player_id', 'player_name']].drop_duplicates(subset='player_id')
            teams = self.SBL.teams(g)
            df_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=777)
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
            events = self.SBL.events(g)
            players = self.SBL.players(g)
            players = players[['player_id', 'player_name']].drop_duplicates(subset='player_id')
            teams = self.SBL.teams(g)
            df_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=777)
            actions = (
            df_actions
            .merge(spadl.actiontypes_df(), how="left")
            .merge(spadl.bodyparts_df(), how="left")
            .merge(players, how="left")
            .merge(teams, how="left")
            )
            all_actions.append(actions)
        df = pd.concat(all_actions)
        return df
    
    def run_pipeline(self):
        comp_season_id = self.sb_get_all_comp_seasons()
        games = self.get_game_ids(comp_season_id)
        if USE_ATOMIC :
            if IMPORT_ALL_DATA:
                # check if file exists
                if not os.path.isfile('sb_all_data_atomic.pkl') or DOWNLOAD_RAW_DATA:
                    df = self.get_all_atomic_actions(games)
                    df.to_pickle('sb_all_data_atomic.pkl')
                else:
                    df = pd.read_pickle("sb_all_data_atomic.pkl")
            else:
                # check if file exists
                if not os.path.isfile('sb_barca_data_atomic.pkl') or DOWNLOAD_RAW_DATA:
                    df = self.get_all_atomic_actions(games)
                    df.to_pickle('sb_barca_data_atomic.pkl')
                else:
                    df = pd.read_pickle("sb_barca_data_atomic.pkl")
        else:
            if IMPORT_ALL_DATA:
                # check if file exists
                if not os.path.isfile('sb_all_data.pkl') or DOWNLOAD_RAW_DATA:
                    df = self.get_all_actions(games)
                    df.to_pickle('sb_all_data.pkl')
                else:
                    df = pd.read_pickle("sb_all_data.pkl")
            else:
                # check if file exists
                if not os.path.isfile('sb_barca_data.pkl') or DOWNLOAD_RAW_DATA:
                    df = self.get_all_actions(games)
                    df.to_pickle('sb_barca_data.pkl')
                else:
                    df = pd.read_pickle("sb_barca_data.pkl")
        return df
    
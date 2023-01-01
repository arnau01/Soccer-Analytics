import os
import socceraction.data.opta as O
import socceraction.spadl as spadl
import socceraction.atomic.spadl as atomicspadl
import pickle
import pandas as pd
from tqdm import tqdm

# Root directory for events data (WhoScored Data to be extracted from zip file)
ROOT = "/Users/arnauayerbe/soccerdata/data/WhoScored/events/"
# Get home teams for each game id
feed = feeds = {'whoscored': "{competition_id}-{season_id}/{game_id}.json",}
opta = O.OptaLoader(root=ROOT,parser="whoscored",feeds=feed)

def create_path(league,season):
    """
    Returns the path and destination for the specified league and season.
    
    Parameters:
    league (str): The league for which to get the data.
    season (str): The season for which to get the data.
    
    Returns:
    Tuple: A tuple containing the path and destination as strings.
    """
    league_options = {
  "ENG": "ENG-Premier League_{season}",
  "ESP": "ESP-La Liga_0910_{season}",
  "FRA": "FRA-Ligue 1_{season}",
  "GER": "GER-Bundesliga_{season}",
  "ITA": "ITA-Serie A_{season}",
  "INT": "INT-World Cup_2223"
    }

    path = league_options[league].format(season=season)
    
    dir = '../pkl_data/Opta/'+league+'/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    destination = dir+'/ws_'+path
    
    return path, destination


# Get all game ids for a given season and league
def get_game_id(path):
    """
    Returns a list of game IDs for the given path.
    
    Parameters:
    path (str): The path to the directory containing the game files.
    
    Returns:
    List: A list of game IDs as strings.
    """
    
    # Get all filenames in folder root+path
    filenames = []
    files = os.listdir(ROOT+path)

    # Iterate through the list of files and append the names to the array
    for file_name in files:
        filenames.append(file_name)
    
    game_ids = []
    for i in filenames:
        # game id is the name of file without .json
        # split by . and take first element
        game_id = i.split('/')[-1].split('.')[0]
        # Check if game id is already in list and is not empty
        if game_id not in game_ids and game_id != '':
            game_ids.append(game_id)
          

    return game_ids

# Get home teams for each game id
def h_teams(game_ids):
    """
    Returns a dictionary mapping game IDs to home teams for the given game IDs.
    
    Parameters:
    game_ids (List[str]): A list of game IDs as strings.
    
    Returns:
    Dict: A dictionary mapping game IDs to home teams as strings.
    """
    
    # Get teams for each game id
    h_teams = []
    print('Getting home teams...')

    for i in range(len(game_ids)):
        df_teams = opta.teams(game_id=game_ids[i])
        
            # if there is a team id column then get the first element and append it
        if 'team_id' in df_teams.columns:
            h_teams.append(opta.teams(game_id=game_ids[i]).team_id[0])
        else:
            # remove the game id from the list
            game_ids.remove(game_ids[i])
    

    # make a dict of game ids and home teams
    game_id_home_team = dict(zip(game_ids, h_teams))

    return game_id_home_team

def events_df(game_ids,game_id_home_team,destination,atomic):

    """
    Returns a dataframe with all SPADL events for the given game IDs, and optionally a dataframe with all atomic SPADL events.
    Also saves the dataframes to pickle files at the specified destination.
    
    Parameters:
    game_ids (List[str]): A list of game IDs as strings.
    game_id_home_team (Dict[str, str]): A dictionary mapping game IDs to home teams as strings.
    destination (str): The file path where the pickle files should be saved.
    atomic (bool): Whether to return a dataframe with atomic SPADL events.
    
    Returns:
    pd.DataFrame: A dataframe with all SPADL events for the given game IDs.
    pd.DataFrame: (optional) A dataframe with all atomic SPADL events for the given game IDs.
    """
    all_data = []
    all_data_atom = []

    print('Getting events...')
    

    for i in tqdm(range(len(game_ids))):
        
        # Get events
        try:
            df = opta.events(game_id=game_ids[i])
        except:
            print('Error getting events for game id: ' + game_ids[i])
            continue
            
        # Convert to SPADL actions
        df = spadl.opta.convert_to_actions(df,home_team_id=game_id_home_team[game_ids[i]])
        if atomic:
            # Convert to atomic actions
            df_atom = atomicspadl.convert_to_atomic(df)

        df = spadl.add_names(df)
        df = spadl.play_left_to_right(df,home_team_id=game_id_home_team[game_ids[i]])
        all_data.append(df)
        

        

        if atomic:
            # Convert to atomic actions
            df_atom = atomicspadl.add_names(df_atom)
            df_atom = atomicspadl.play_left_to_right(df_atom,home_team_id=game_id_home_team[game_ids[i]])
            all_data_atom.append(df_atom)

    # Concatenate all dataframes
    events = pd.concat(all_data)
    if atomic:
        events_atom = pd.concat(all_data_atom)

    # Save to pickle
    events.to_pickle(destination+'_spadl_test.pkl')
    if atomic:
        events_atom.to_pickle(destination+'_atom_test.pkl')

# Run loader 
def run(league,season,atomic=False):
        """
        Run the pipeline to extract and transform data from WhoScored for a given league and season.

        Args:
        league (str): The league to extract data for. Must be one of 'ENG', 'ESP', 'FRA', 'GER', 'ITA', 'INT'.
        season (str): The season to extract data for. Must be in the format 'YYYY' where Y is a digit.
        atomic (bool, optional): Whether to also extract and transform data into atomic actions. Defaults to False.

        Returns:
        None
        """
        path, destination = create_path(league, season)
        # Get game ids
        game_ids = get_game_id(path)
        # Get home teams
        game_id_home_team = h_teams(game_ids)
        # Get events data
        events_df(game_ids, game_id_home_team, destination, atomic)

if __name__ == '__main__':
    
    run(league='FRA',season='1415')
   
    
    

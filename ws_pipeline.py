import os
import soccerdata as sd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl


# Step 1: Initialise the scraper

# Available leagues : 
# ['ENG-Premier League',
#  'ESP-La Liga',
#  'FRA-Ligue 1',
#  'GER-Bundesliga',
#  'INT-World Cup',
#  'ITA-Serie A']

# Can be single league or list of leagues
league = "ENG-Premier League"
# Can be single season or list of seasons
season = [12]
id = league[:3] + str(season)
ws = sd.WhoScored(leagues=league, seasons=season)

# Step 2: Get the events match data
def get_events():
    # if USE_ATOMIC:
    events_atom = ws.read_events(output_fmt="atomic-spadl")
    events_atom = atomicspadl.add_names(events_atom)

    # else:
    events = ws.read_events(output_fmt="spadl")
    events = spadl.add_names(events)
    
    return events,events_atom
    

# Fetch dataframes
if __name__ == '__main__':
    print("Fetching data for " + id)
    events,events_atom = get_events()
    
    # Create directory inside the pkl_data directory if it doesn't exist called ws
    if not os.path.exists('../pkl_data/ws'):
        os.makedirs('../pkl_data/ws')

    # events.to_pickle('../pkl_data/ws/'+filename)
    events.to_pickle('../pkl_data/ws/ws_'+id+'_data.pkl')
    events_atom.to_pickle('../pkl_data/ws/ws_'+id+'_atomic.pkl')
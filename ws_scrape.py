import os
import soccerdata as sd
import socceraction.atomic.spadl as atomicspadl
import socceraction.spadl as spadl
    

# Fetch dataframes
if __name__ == '__main__':


    seasons = ['2022-2023']
    
    leagues = ['ENG-Premier League','ITA-Serie A']

    # leagues = ['FRA-Ligue 1']
    for i in range(len(leagues)):
        league = leagues[i]
        for j in range(len(seasons)):
            
            season = seasons[j]
            
            dir = '../pkl_data/ws/'+league[:3]+'/'
            id = league[:3] + str(season[2:4])
            if not os.path.exists(dir):
                os.makedirs(dir)
            ws = sd.WhoScored(leagues=league, seasons=season)

            
            print(id)
            ws = sd.WhoScored(leagues=league, seasons=season)
            

            # else:
            events = ws.read_events(output_fmt="spadl")
            events = spadl.add_names(events)
            
            # Create directory inside the pkl_data directory if it doesn't exist called ws
            

            # events.to_pickle('../pkl_data/ws/'+filename)
            events.to_pickle(dir+'ws_'+id+'_data.pkl')
            

        
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

import bin_action_seq as bs

# Find similar sequences
# Generate a heatmap for the set of similar actions
# Then find what happens in all the next actionspip i



warnings.filterwarnings('ignore')
# Amount of bins to adjust heatmap
X_B = bs.X_B
Y_B = bs.Y_B
M = bs.M
REBUILD_DATA = bs.REBUILD_DATA
# Amount of start actions to check similarity
ST = 4
bin_data = bs.file_name

def load_data():
    
    action_data = np.load(bin_data, allow_pickle=True)
    seq_names = action_data.files
    # print(len(seq_names))
    for name in seq_names:
        # print(name)
        np_data = action_data[name]
        # print(np_data)
    return np_data

def find_sim(np_data):
    # Get second dimesion of array (bin number)
    b = np.delete(np_data, [0,2], axis=2)
    b = b.reshape((b.shape[0], b.shape[1] * b.shape[2]))
    df_b = pd.DataFrame(b)

    # Get (x,y) tuple for heatmap
    x_y = np.delete(np_data, [0,1], axis=2)
    x_y = x_y.reshape((x_y.shape[0], x_y.shape[1] * x_y.shape[2]))
    df_xy = pd.DataFrame(x_y)

    df_start = df_b.iloc[:, :ST]
    # Get matching indices of duplicate rows
    # df_start = df_start[df_start.duplicated(keep=False)]
    matches = df_start.groupby(list(df_start)).apply(lambda x: tuple(x.index)).tolist()
   
    # Sort list by length
    matches.sort(key=len, reverse=True)

    return matches,df_xy
    
def generate_hm(index,df_xy):
    # df_match is the filtered df for matches

    df_match = df_xy.iloc[df_xy.index.isin(index)]


    # def generate_heatmap(np_data,index,name):
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
    fig, ax = pitch.draw()


    # Section to get x and y coordinates from df

    start_seq = df_match.iloc[0, :8]
    # Get x bin for all the actions, the x bin is the first number in the tuple
    x = pd.Series([i[0] for i in start_seq])*(105/X_B)
    y = pd.Series([i[1] for i in start_seq])*(68/Y_B)


    # Convert our sequence to the bin_statistic format
    stats = pitch.bin_statistic(x, y,bins = (X_B,Y_B), statistic='count')
    # print(start_seq[0])
    # # Plot start_seq as a heatmap in pitch
    pcm = pitch.heatmap(stats, cmap='plasma', ax=ax)
    pcm = pitch.heatmap(stats, cmap='plasma', ax=ax)
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
    cbar.outline.set_edgecolor('#efefef')
    cbar.outline.set_linewidth(2)
    # save the figure
    dir_name = "./HM/{}/{}/{}/{}_{}/".format(X_B*Y_B,ST,bs.USE_ATOMIC,len(index),index[0])
    results_dir = os.path.join(dir_name)
    sample_file_name = "start_seq.png"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)

    for i in range(ST,len(df_match.columns)):
        
        
        # For each subsequent action get all the x and y coordinates
        x_i = pd.Series([i[0] for i in df_match.iloc[:, i]])*(105/X_B)
        y_i = pd.Series([i[1] for i in df_match.iloc[:, i]])*(68/Y_B)
        # Convert our sequence to the bin_statistic format
        stats = pitch.bin_statistic(x_i, y_i,bins = (X_B,Y_B), statistic='count')
        # print(start_seq[0])
        # # Plot start_seq as a heatmap in pitch
        pcm = pitch.heatmap(stats, cmap='plasma', ax=ax)
        pcm = pitch.heatmap(stats, cmap='plasma', ax=ax)
        # cbar = fig.colorbar(pcm, ax=ax, shrink=0.6)
        # cbar.outline.set_edgecolor('#efefef')
        # cbar.outline.set_linewidth(2)
        # save the figure
        sample_file_name = "start_seq_+{}.png".format(i+1)
        
        fig.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)

    


def generate_kde(index,df_xy):

    plt.clf()

    df_match = df_xy.iloc[df_xy.index.isin(index)]
    # Plot a kernel density estimate of locations of shots
    # mps.field("white",figsize=8, show=False)
    pitch = Pitch(pitch_type='statsbomb', line_zorder=2, pitch_color='#22312b', line_color='#efefef')
    fig, ax = pitch.draw()
    start_seq = df_match.iloc[0, :8]
    # Get x bin for all the actions, the x bin is the first number in the tuple
    x = pd.Series([i[0] for i in start_seq])*(105/X_B)
    y = pd.Series([i[1] for i in start_seq])*(68/Y_B)
    # create a dataframe with the x and y coordinates
    df = pd.DataFrame({'x': x, 'y': y})
    
    sns.kdeplot(data=df,x='x',y='y', shade=True, shade_lowest=False, cmap="plasma")
    # sns.kdeplot(df_s['x'], df_s['y'],shade=True,cmap="plasma")
    plt.axis("on")
    # plt.show()
    # save the figure
    dir_name = "./HM/{}/{}/{}/{}_{}/".format(X_B*Y_B,ST,bs.USE_ATOMIC,len(index),index[0])
    results_dir = os.path.join(dir_name)
    sample_file_name = "kde_start_seq.png"
    
    plt.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    for i in range(ST,len(df_match.columns)):
        # Clear sns
        plt.clf()
        pitch = Pitch(pitch_type='opta',pitch_length=105, pitch_width=68, line_zorder=0, pitch_color='#22312b', line_color='#efefef')
        fig, ax = pitch.draw()
        # For each subsequent action get all the x and y coordinates
        x_i = pd.Series([i[0] for i in df_match.iloc[:, i]])*(105/X_B)
        y_i = pd.Series([i[1] for i in df_match.iloc[:, i]])*(68/Y_B)
        
        df = pd.DataFrame({'x': x_i, 'y': y_i})
        sns.kdeplot(data=df,x='x',y='y', shade=True, shade_lowest=False, cmap="plasma")

        
        sample_file_name = "kde_start_seq_+{}.png".format(i+1)
        
        plt.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)


def generate_charts(matches):

    plt.clf()
    # Plot a bar chart of the frequency of each length of similar sequences
    print(len(matches))
    # Get the length of each list in matches
    lengths = [len(x) for x in matches]
    # Get the unique lengths and their frequency
    unique_lengths, counts = np.unique(lengths, return_counts=True)
    # Plot the bar chart
    plt.bar(unique_lengths, counts)
    plt.title("Bar chart of amount similar sequences on {} pitch with initial Seq. Len. : {}, Atomic: {}".format(X_B*Y_B,ST,bs.USE_ATOMIC))
    plt.xlabel("Amount of similar sequences")
    plt.ylabel("Frequency")
    plt.ylim(0, 500)
    plt.savefig("./HM/{}/{}/bar.png".format(X_B*Y_B,ST,bs.USE_ATOMIC), dpi=300, bbox_inches='tight', pad_inches=0.1)


    # Calculate percentage 
    total = len(matches)
    # make percent accurate to 4 decimal places or 2 sig figs
    percent = [round(((x*100)/total),4) for x in counts]

    # print(percent)
    

    plt.clf()

    fig = plt.hist([len(x) for x in matches], bins=len(matches[0]))
    plt.title("Histogram of amount similar sequences on {} pitch with initial Seq. Len. : {}, Atomic: {}".format(X_B*Y_B,ST,bs.USE_ATOMIC))
    plt.xlabel("Amount of similar sequences")
    plt.ylabel("Frequency")
    plt.ylim(0, 500)
    plt.savefig("./HM/{}/{}/hist.png".format(X_B*Y_B,ST,bs.USE_ATOMIC), dpi=300, bbox_inches='tight', pad_inches=0.1)

    plt.clf()

    # Create a table of the unique lengths and their frequency
    table = pd.DataFrame({'Length':unique_lengths, 'Frequency':counts, 'Percentage':percent})
    # Save the table as a plt figure
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=table.values, colLabels=table.columns, loc='center')

    fig.tight_layout()
    plt.savefig("./HM/{}/{}/table.png".format(X_B*Y_B,ST,bs.USE_ATOMIC), dpi=300, bbox_inches='tight', pad_inches=0.1)

if __name__ == '__main__':

    # If file name exists, load it
    if os.path.exists(bin_data) and not REBUILD_DATA:
        np_data = load_data()
    
    # If bin file doesn't exist, create it
    else:
        # Call main method from bin_action_seq.py
        bs.main()
        np_data = load_data()
    
    print("Finding similar sequences")
    # Add conditions like attacking half
    matches, df_xy = find_sim(np_data)
    

    if len(matches) == 0:
        print("No similar sequences")
    # Generate heatmap for similar sequences

    for i in range(len(matches[:3])) :        
        generate_hm(matches[i],df_xy)
        generate_kde(matches[i],df_xy)
    
    generate_charts(matches)
    
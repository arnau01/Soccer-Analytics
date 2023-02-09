#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotsoccer as mps
from mplsoccer import Pitch
TOTAL=1000

def create_data(l=3,t_fixed=True,noise_size=1):
    noise_size = int(noise_size * 100)
    n = int(TOTAL - noise_size)
    print(n)
    if t_fixed:
        data = np.array([[10,55,10],[25,55,60],[65,10,35]])
        # Make sure sequence is of correct length
        while len(data) < l:
            # time -> last time + 5
            # x -> between 0 and 105
            # y -> between 0 and 68
            time = data[-1,0] + 5
            x = np.random.uniform(0, 105)
            y = np.random.uniform(0, 68)
            data = np.vstack((data, [time, x, y]))

        all_data = np.stack([data]*n)

    # Sequences occur at random times between 0 and 45
    # The first column (time) between 0 and 45
    # The first column has to be in ascending order
    # The second column (x) is still fixed at 55,55,10
    # The third column (y) is still fixed at 10,60,35
    
    else: 
        all_data = np.empty((n, l, 3))
        for i in range(n):
            time = np.sort(np.random.uniform(0, 45, l))
            x = np.array([55,55,10])
            # Fill to correct length
            while len(x) < l:
                x = np.append(x, np.random.uniform(0, 105))
            y = np.array([10,60,35])
            # Fill to correct length
            while len(y) < l:
                y = np.append(y, np.random.uniform(0, 68))
            all_data[i] = np.column_stack((time, x, y))


    # Add a bit of noise to the data
    # With the same format as the original data
    # The first column (time) between 0 and 45
    # The first column has to be in ascending order
    # The second column (x) between 0 and 105
    # The third column (y) between 0 and 68
    # Create 1000 random arrays of shape (3,3)

    np.random.seed(0)  # set seed for reproducibility

    
    print(noise_size)
    noise = np.empty((noise_size, l, 3))  # empty array to store the arrays
    # Generate noise_size arrays
    for i in range(noise_size):
        # Generate random times between 0 and 45 in ascending order
        time = np.sort(np.random.uniform(0, 45, l))
        # Generate random x between 0 and 105
        x = np.random.uniform(0, 105, l)
        # Generate random y between 0 and 68
        y = np.random.uniform(0, 68, l)
        # Stack the arrays horizontally
        noise[i] = np.column_stack((time, x, y))


        # print(noise)
        # print(all_data)


    # Stack arrays vertically
    all_data = np.vstack((all_data, noise))
    print(len(all_data))


    # shuffle the data
    np.random.shuffle(all_data)
    print(all_data)
    filename = "interim/"+str(TOTAL)+str(t_fixed)+".npz"
    np.savez(filename, arr_0=all_data)
    return filename
#%%
def display(data):
    mps.field("green",figsize=8, show=False)
    data = data[:,:,1:]
    # x is the first column of each array
    x = data[:,:,0]
    # y is the second column of each array
    y = data[:,:,1]
    plt.scatter(x,y)
    plt.axis("on")
    plt.show()

    pitch = Pitch(line_color='#000009', line_zorder=2,pitch_length=105,pitch_width=68)
    fig, ax = pitch.draw()
    kde = pitch.kdeplot(x, y, ax=ax,
                    # fill using 100 levels so it looks smooth
                    fill=True, levels=100,
                    # shade the lowest area so it looks smooth
                    # so even if there are no events it gets some color
                    shade_lowest=True,
                    cut=4,  # extended the cut so it reaches the bottom edge
                    cmap='plasma')

# Create main to run the code
if __name__ == '__main__':
    filename = create_data(l=3,t_fixed=True,noise_size=1)
    loaded = np.load(filename,allow_pickle=True)
    barca = np.load("interim/data_seq_barca.npz",allow_pickle=True)
    data = loaded["arr_0"]
    barca = barca["arr_0"]
    # stack two arrays vertically
    # data = np.vstack((data,barca))
    print(data)
    display(data)
    # shuffle the data
    np.random.shuffle(data)
    # Take only 1000 first elements
    # data = data[:1000]
    # Save the data
    np.savez("interim/data_seq_socc.npz", arr_0=data)

# %%

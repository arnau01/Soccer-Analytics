#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotsoccer as mps
from mplsoccer import Pitch
TOTAL=10000

# Generate synthetic data with gaussian distribution
def gaussian_data(l=6,t=[0,2,5,9,14,20,26,33],x=[10,20,30,60,90,100,70,80],y=[8,60,10,60,10,60,56,64],t_fixed=True,):

    n = int(TOTAL)
    # Make data length
    t = t[:l]
    x = x[:l]
    y = y[:l]

    all_x = []
    all_y = []

    x_std = .1
    y_std = x_std * 68 / 105

    # Create gaussian distribution
    for i in range(l):
        x1 = np.random.normal(x[i], x_std*(i+1), TOTAL)
        y1 = np.random.normal(y[i], y_std*(i+1), TOTAL)
        # Put all x in a column
        all_x.append(x1.reshape(-1, 1))
        # Put all y in a column
        all_y.append(y1.reshape(-1, 1))

    all_data = np.empty((n, l, 3))
    # Fill the first column of each array with the time
    all_data[:, :, 0] = t
    
    # Concatenate all the x arrays in all_x
    x = np.concatenate(all_x, axis=1)
    # make sure they're less than 105
    x[x > 105] = 105
    # make sure they're greater than 0
    x[x < 0] = 0
    all_data[:, :, 1] = x
    # Concatenate all the y
    y = np.concatenate((all_y), axis=1)
    # make sure they're less than 68 in value
    y[y > 68] = 68
    # make sure they're greater than 0 in value
    y[y < 0] = 0
    all_data[:, :, 2] = y

    return all_data

def rand_noise(noise_size=int(TOTAL/4),l=6):
    noise = np.empty((noise_size, l, 3))  # empty array to store the arrays
    # Generate noise_size arrays
    
    for i in range(noise_size):
        # Generate random times between 0 and 45 in ascending order
        t0 = 0
        time = np.sort(np.random.uniform(0, 15, l-1))
        # Add t0 to the beginning of the array
        time = np.insert(time, 0, t0)
        # Generate random x between 0 and 105
        x = np.random.uniform(0, 105, l)
        # Generate random y between 0 and 68
        y = np.random.uniform(0, 68, l)
        # Stack the arrays horizontally
        noise[i] = np.column_stack((time, x, y))
        # Add the array to the list of arrays


    return noise
#%%
def display(data,color="blue"):
    mps.field("green",figsize=8, show=False)
    data = data[:,:,1:]
    # drop the 4th row of each array
    # data = data[:, :-1, :]
    # x is the first column of each array
    x = data[:,:,0]
    # y is the second column of each array
    y = data[:,:,1]
    # Subtract 68 from y to flip the y axis
    l = data.shape[1]
    print(l)
    # plt.scatter(x,y,color=color)
    data1 = data[:, :int(l/2), :]
    data2 = data[:, int(l/2):, :]
    x1 = data1[:,:,0]
    y1 = data1[:,:,1]
    plt.scatter(x1,y1,color="black")
    x2 = data2[:,:,0]
    y2 = data2[:,:,1]
    plt.scatter(x2,y2,color="red")
    plt.axis("on")
    plt.show()
    # y = 68 - y
    # pitch = Pitch(line_color='#000009', line_zorder=2,pitch_length=105,pitch_width=68,pitch_type='uefa')
    # fig, ax = pitch.draw()
    # kde = pitch.kdeplot(x, y, ax=ax,
    #                 # fill using 100 levels so it looks smooth
    #                 fill=True, levels=100,
    #                 # shade the lowest area so it looks smooth
    #                 # so even if there are no events it gets some color
    #                 shade_lowest=True,
    #                 cut=4,  # extended the cut so it reaches the bottom edge
    #                 cmap='plasma')

# Create main to run the code
if __name__ == '__main__':
    
    l = 2
    m = int(l/2)

    # Make t a array of l numbers increasing 1+i each time (0,1,3,6,10,15)
    #t = np.cumsum(np.arange(0, l))

    # t = np.arange(0, l)
    # Make an array of a zero and then (l-1) 1's
    t = np.concatenate((np.zeros(1), np.ones(l-1))) 
    # set x, 10 points which are 10 and 10 points which are 20
    x = np.concatenate((np.full(1, 80),np.full(1,100)))
    # set y, 10 points which are 8 and 10 points which are 60
    y = np.concatenate((np.full(1, 60),np.full(1,34)))
    data = gaussian_data(l=l,t=t,x=x,y=y)

    # r = rand_noise(noise_size=1000,l=l)
    # data = np.concatenate((data,r))
    display(data,color="black")
    
    # # set x, 10 points which are 10 and 10 points which are 20
    # x = np.concatenate((np.full(10, 30), np.full(10, 70)))
    # # set y, 10 points which are 8 and 10 points which are 60
    # y = np.concatenate((np.full(10, 30), np.full(10, 10)))
    # data2 = gaussian_data(l=20,t=t,x=x,y=y)
    # display(data2,color="green")

    # # set x, 10 points which are 10 and 10 points which are 20
    # x = np.concatenate((np.full(10, 10), np.full(10, 40)))
    # # set y, 10 points which are 8 and 10 points which are 60
    # y = np.concatenate((np.full(10, 10), np.full(10, 60)))
    # data3 = gaussian_data(l=20,t=t,x=x,y=y)
    # display(data3,color="green")

    # data = np.vstack((data, data2,data3))

    # Stack arrays on top of each other
    # data = np.vstack((data, noise))
    # display(data,color="black")
    # Save data to .npz
    np.savez("interim/data_seq_socc.npz", arr_0=data)

    

# %%

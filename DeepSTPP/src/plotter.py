import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plotly.graph_objects as go
from tqdm.auto import tqdm
from mplsoccer import Pitch
from datetime import datetime
import os



"""
outputs: [batch, lookahead, 3] or [batch, 3]
targets: [batch, lookahead, 3] or [batch, 3]
"""
def visualize_diff(outputs, targets, portion=1):
    if len(targets.shape) == 2:
        outputs = np.expand_dims(outputs, 1)
        targets = np.expand_dims(targets, 1)

    outputs = outputs[:int(len(outputs) * portion)]
    targets = targets[:int(len(targets) * portion)]

    plt.figure(figsize=(14, 10), dpi=180)
    plt.subplot(2, 2, 1)

    n = outputs.shape[0]
    lookahead = outputs.shape[1]

    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 0], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 0], "-o", color="b", label="Actual")
    plt.ylabel('Latitude')
    plt.legend()

    plt.subplot(2, 2, 2)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 1], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 1], "-o", color="b", label="Actual")
    plt.ylabel('Longitude')
    plt.legend()

    plt.subplot(2, 2, 3)
    for i in range(lookahead):
        plt.plot(range(i, n), outputs[:n - i, i, 2], "-o", label=f"Predicted {i} step")
    plt.plot(targets[:, 0, 2], "-o", color="b", label="Actual")
    plt.ylabel('delta_t (hours)')
    plt.legend()
    plt.savefig('result.png')
    
    
def frame_args(duration):
    return {
               "frame": {"duration": duration},
               "mode": "immediate",
               "fromcurrent": True,
               "transition": {"duration": duration},
           }


"""
x_range, y_range, t_range: 1D array of any length
"""
def inverse_transform(x_range, y_range, t_range, scaler):
    # Inverse transform the data
    temp = np.zeros((len(x_range), 3)) 
    temp[:, 0] = x_range
    x_range = scaler.inverse_transform(temp)[:, 0]

    temp = np.zeros((len(y_range), 3)) 
    temp[:, 1] = y_range
    y_range = scaler.inverse_transform(temp)[:, 1]

    temp = np.zeros((len(t_range), 3)) 
    temp[:, 2] = t_range
    t_range = scaler.inverse_transform(temp)[:, 2]
    
    return x_range, y_range, t_range


"""
lambs: list, len(lambs) = len(t_range), element: [len(x_range), len(y_range)]
fps: # frame per sec
fn: file_name

The result could be saved as file with command `ani.save('file_name.mp4', writer='ffmpeg', fps=fps)`
                                    or command `ani.save('file_name.gif', writer='imagemagick', fps=fps)`
"""
def plot_lambst_static(lambs, x_range, y_range, t_range, fps, scaler=None, cmin=None, cmax=None, 
                       history=None, decay=0.3, base_size=300, cmap='magma', fn='result.mp4'):
    # Inverse transform the range to the actual scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)
        
    if cmin is None:
        cmin = 0
    if cmax is "outlier":
        cmax = np.max([np.max(lamb_st) for lamb_st in lambs])
    if cmax is None:
        cmax = np.max(lambs)
    print(f'Inferred cmax: {cmax}')
    cmid = cmin + (cmax - cmin) * 0.9
        
    grid_x, grid_y = np.meshgrid(x_range, y_range)
    
    frn = len(t_range) # frame number of the animation
    
    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = fig.add_subplot(111, projection='3d', xlabel='x', ylabel='y', zlabel='λ', zlim=(cmin, cmax), 
                         title='Spatio-temporal Conditional Intensity')
    ax.title.set_position([.5, .95])
    text = ax.text(min(x_range), min(y_range), cmax, "t={:.2f}".format(t_range[0]), fontsize=10)
    plot = [ax.plot_surface(grid_x, grid_y, lambs[0], rstride=1, cstride=1, cmap=cmap)]
    
    if history is not None: 
        his_s, his_t = history
        zs = np.ones_like(lambs[0]) * cmid # add a platform for locations
        plat = ax.plot_surface(grid_x, grid_y, zs, rstride=1, cstride=1, color='white', alpha=0.2)
        points = ax.scatter3D([], [], [], color='black') # add locations 
        plot.append(plat)
        plot.append(points)
    
    pbar = tqdm(total=frn + 2)
    
    def update_plot(frame_number):
        t = t_range[frame_number]
        plot[0].remove()
        plot[0] = ax.plot_surface(grid_x, grid_y, lambs[frame_number], rstride=1, cstride=1, cmap=cmap)
        text.set_text('t={:.2f}'.format(t))
        
        if history is not None:
            mask = np.logical_and(his_t <= t, his_t >= t_range[0])
            locs = his_s[mask]
            times = his_t[mask]
            sizes = np.exp((times - t) * decay) * base_size
            zs = np.ones_like(sizes) * cmid
            plot[2].remove()
            plot[2] = ax.scatter3D(locs[:, 0], locs[:, 1], zs, c='black', s=sizes, marker='x')
        
        pbar.update()
    
    ani = animation.FuncAnimation(fig, update_plot, frn, interval=1000/fps)
    ani.save(fn, writer='ffmpeg', fps=fps)
    return ani
    
    
"""
lambs: list, len(lambs) = len(t_range), element: [len(x_range), len(y_range)]
"""
def plot_lambst_interactive(lambs, x_range, y_range, t_range, cmin=None, cmax=None, 
                            scaler=None, heatmap=False):

    # POTENTIAL ISSUE HERE -> x_raange is set to 105
    # Scaler may not be correct


    # Inverse transform the range to the actual scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)
    
    if cmin is None:
        cmin = 0
    if cmax is "outlier":
        cmax = np.max([np.max(lamb_st) for lamb_st in lambs])
    if cmax is None:
        cmax = np.max(lambs)
    frames = []

    for i, lamb_st in enumerate(lambs):
        if heatmap:
            frames.append(go.Frame(data=[go.Heatmap(z=lamb_st, x=x_range, y=y_range, zmin=cmin, 
                                                    zmax=cmax)], name="{:.2f}".format(t_range[i])))
        else:
            frames.append(go.Frame(data=[go.Surface(z=lamb_st, x=x_range, y=y_range, cmin=cmin, 
                                                    cmax=cmax)], name="{:.2f}".format(t_range[i])))
    
    fig = go.Figure(frames=frames)
    
    # Add data to be displayed before animation starts
    if heatmap:
        fig.add_trace(go.Heatmap(z=lambs[0], x=x_range, y=y_range, zmin=cmin, zmax=cmax))
    else:
        fig.add_trace(go.Surface(z=lambs[0], x=x_range, y=y_range, cmin=cmin, cmax=cmax))

    # Slider
    sliders = [
                  {
                       "pad": {"b": 10, "t": 60},
                       "len": 0.9,
                       "x": 0.1,
                       "y": 0,
                       "steps": [
                           {
                               "args": [[f.name], frame_args(0)],
                               "label": f.name,
                               "method": "animate",
                           }
                           for f in fig.frames
                       ],
                   }
               ]
    
    # Layout
    fig.update_layout(
        title='Spatio-temporal Conditional Intensity',
        width=600,
        height=600,
        scene=dict(
                   zaxis=dict(range=[cmin, cmax], autorange=False),
                   aspectratio=dict(x=1, y=1, z=1),
                  ),
        updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(1)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
        sliders=sliders
    )
    fig.show()


def plot_lambst_on_football_pitch(lambs, x_range, y_range, t_range, data,lookback,forward,cmin=None, cmax=None, scaler=None):
    # Create the football field
    pitch = Pitch(line_color='#000009', line_zorder=2,pitch_length=105,pitch_width=68,pitch_type='uefa')
    # specifying figure size (width, height)
    
    fig, ax = pitch.draw()
    

    x = data[:,:,1]
    # y is the second column of each array
    y = data[:,:,2]
    t = data[:,:,0][0]

    # Get the first lookback columns
    x_back = x[:,:lookback]
    y_back = y[:,:lookback]

    x_for = x[:,lookback:lookback+forward]
    y_for = y[:,lookback:lookback+forward]


    
    
    t_pred = t[lookback:]
    # Find the index in lambs which is closest to t_pred
    # In t_range get the index which is closest or ideally equal to t_pred
    # For each t_pred there is a corresponding lamb_st
    idx=[]
    for i in range(len(t_pred)):
        idx.append(np.abs(t_range - t_pred[i]).argmin())
        
    # Flip lambs to have the correct orientation
    lambs = np.flip(lambs, axis=1)

    # Scatter of first points in color black
    plt.scatter(x_back, y_back, color='black') 
    # Draw an arrow -> from the first point to the second point 
    for i in range(lookback-1):
        ax.annotate("",
                xy=(x[0][i+1], y[0][i+1]), xycoords='data',
                xytext=(x[0][i], y[0][i]), textcoords='data',
                
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=0",
                                lw=2.5,
                                shrinkA=20+(i*2), shrinkB=20+(i*2),
                                ),
                )
    
    # # Scatter of last two points in color red
    plt.scatter(x_for, y_for, color='red')
    # # Draw an arrow from the second to last point to the last point with direction
    for i in range(forward):
        ax.annotate("",
                xy=(x[0][lookback+i], y[0][lookback+i]), xycoords='data',
                xytext=(x[0][lookback-1+i], y[0][lookback-1+i]), textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                connectionstyle="arc3,rad=0",
                                lw=2.5,
                                shrinkA=30+(i*2), shrinkB=30+(i*2),
                                color='red'
                                ),
            )
    
    # Inverse transform the range to the actual scale
    if scaler is not None:
        x_range, y_range, t_range = inverse_transform(x_range, y_range, t_range, scaler)
    
    # Set color limits for the heatmap
    if cmin is None:
        cmin = 0
    if cmax is "outlier":
        cmax = np.max([np.max(lamb_st) for lamb_st in lambs])
    if cmax is None:
        cmax = np.max(lambs)
    
    
    
    # Plot the heatmap for each idx
    # for i in range(forward):
    #     # Make lambs the sum of all the lamb_idx
    #     lambs_cum = np.sum(lambs[idx], axis=0)
        
        # Add a colorbar to the right of the field
        

    # im = ax.imshow(lambs[idx], cmap='plasma', vmin=cmin, vmax=cmax,extent=(0, 105, 0, 68))
    im = ax.imshow(lambs[idx[forward-1]], cmap='plasma', vmin=cmin, vmax=cmax,extent=(0, 105, 0, 68))
    # Add a colorbar to the right of the field
    cbar = fig.colorbar(im, ax=ax, orientation='vertical')
    
    # Show the plot
    # plt.show()
    # Save the plot
    now = datetime.now()
    # if directory does not exist, create it
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    fig.savefig('predictions/pred-'+now.strftime("%d_%H-%M")+'.png', dpi=300, bbox_inches='tight')

    
    

class TrajectoryPlotter:
    
    def __init__(self):
        self.data = []
        self.layout = go.Layout(
            width  = 1200,
            height = 600,
            scene  = dict(
                camera=dict(
                    up  = dict(x=1, y=0., z=0),
                    eye = dict(x=0., y=2.5, z=0.)
                ),
                xaxis = dict(title="latitude"),
                yaxis = dict(title="longitude"),
                zaxis = dict(title="time"), 
                aspectmode  = "manual",
                aspectratio = dict(x=1, y=1, z=3)
            ),
            showlegend = True,
        )
        
    
    """
    outputs: [batch, lookahead, 3] or [batch, 3]
    targets: [batch, lookahead, 3] or [batch, 3]
    """
    def compare(self, outputs, targets):
        if len(targets.shape) == 2:
            outputs = np.expand_dims(outputs, 1)
            targets = np.expand_dims(targets, 1)
        
        target_t = np.append(0, np.cumsum(targets[:, 0, 2]))
        self.add_trace(targets[:, 0, 0], targets[:, 0, 1], target_t, "actual")
        
        n = outputs.shape[0]
        lookahead = outputs.shape[1]
        for i in range(lookahead):
            output_t = np.append(0, np.append(0, np.cumsum(targets[:n-i-1, 0, 2])) + outputs[:n-i, i, 2])
            self.add_trace(outputs[:n-i, i, 0], outputs[:n-i, i, 1], output_t, f"Predicted {i} step")
        
    
    """
    x, y, z: [batch]
    """
    def add_trace(self, x, y, z, name=None, color=None):
        self.data.append(go.Scatter3d(
                             x = x,
                             y = y,
                             z = z,
                             name = name,
                             mode = 'lines+markers',
                             marker = dict(
                                 size   = 4,
                                 symbol = 'circle',
                                 color  = color,
                             ),
                             line = dict(
                                 width  = 3,
                                 color  = color,
                             ),
                             opacity = .6
                        ))
        
        
    def show(self):
        fig = go.Figure(data=self.data, layout=self.layout)
        fig.show()

## WiP - not working yet 
# # Seeing how gamma changes with time

# from model import s_intensity
# # Function to plot spatial intensity heatmap
# def plot_s_intensity(w_i, b_i, t_ti, inv_var, x_range=[0, 105], y_range=[0, 68]):
#     # Define the spatial grid over which to plot the heatmap
#     x_grid, y_grid = np.meshgrid(np.arange(x_range[0], x_range[1]+1), np.arange(y_range[0], y_range[1]+1))

#     # Compute the spatial intensity at each grid point
#     s_diff = torch.tensor(np.concatenate([x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)], axis=1))
#     spatial_intensity = s_intensity(w_i, b_i, t_ti, s_diff, inv_var)
#     spatial_intensity = spatial_intensity.reshape(len(y_grid), len(x_grid))

#     # Plot the heatmap using matplotlib
#     fig, ax = plt.subplots()
#     im = ax.imshow(spatial_intensity, cmap='plasma', origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     plt.colorbar(im)
#     plt.show()
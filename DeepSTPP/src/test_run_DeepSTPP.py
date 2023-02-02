#%%
import numpy as np
# import pandas as pd
# import sys
#
# from copy import deepcopy
# from tqdm.auto import tqdm, trange
#
# from IPython.display import SVG, display
import plotly.graph_objects as go
# import plotly.express as px
#
# # from data.synthetic import *
# from plotter import *
import os

from argparse import ArgumentParser
import os
import sys
import random
import logging
import numpy as np

from scipy import integrate
from sklearn.metrics import mean_squared_error as MSE
import copy

from tqdm import tqdm, trange
# from tqdm.contrib import tenumerate

import torch
from torch.utils.data import DataLoader

BATCH_SIZE = 128 #32
N_EPOCHS = 5 #250
EVAL_EPOCHS = 5
NUM_BACKGROUND_POINTS = 0
TOTAL_TIME = 30.0
RELOAD_DATA = False
TRAIN_MODEL = True
TRAIN_RATIO = 0.9 #0.9
VAL_RATIO = 0.05 #0.05
TEST_RATIO = 0.05 #0.05
#%%

def summarize(data):
    print(f'number of data: {len(data)}')

    seq_lens = [len(seq) for seq in data]
    print(f'sequence length range: {min(seq_lens)} ~ {max(seq_lens)}')

    ranges = []
    for i in range(3):
        start = min([seq[0, i] for seq in data])
        end = max([seq[-1, i] for seq in data])
        ranges.append((start, end))

    print(f'time range: {ranges[0][0]:.3f} ~ {ranges[0][1]:.3f}')
    print(f's1 range:   {ranges[1][0]:.3f} ~ {ranges[1][1]:.3f}')
    print(f's2 range:   {ranges[2][0]:.3f} ~ {ranges[2][1]:.3f}')

    fig = go.Figure(data=[go.Histogram(x=seq_lens, histnorm='probability', name='Sequence Lengths')])
    fig.show()


# split dataset into train, validation, test
def split_dataset(seq, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1

    # split seqs
    n = len(seq)
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n

    train_seq = seq[:train_n]
    val_seq = seq[train_n:train_n + val_n]
    test_seq = seq[train_n + val_n:]

    return train_seq, val_seq, test_seq


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

#%%

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # set max_split_size_mb
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4000'
    if torch.cuda.is_available():
        print("You are using GPU acceleration.")
        torch.cuda.set_per_process_memory_fraction(0.8, device=None)


    sys.path.append("src")

    cwd = os.getcwd()
    print(cwd)

    # from plotter import *
    from model import *
    from data.dataset import SlidingWindowWrapper
    from util import *


    #check file exists
    if not os.path.exists('data/processed/data_seq.npz') or RELOAD_DATA == True:

        dataset = np.load('data/interim/data_seq.npz')

        dataset = dataset['arr_0']
        train_seq, val_seq, test_seq = split_dataset(dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                                                     test_ratio=TEST_RATIO)

        file_splits = {"train": train_seq, "val": val_seq, "test": test_seq}
        for key, value in file_splits.items():
            print(f'{key} set contains {len(value)} sequences')

        # if directory does not exist, create it
        if not os.path.exists('data/processed'):
            os.makedirs('data/processed')

        with open('data/processed/data_seq.npz', 'wb') as f:
            np.savez_compressed(f,
                     train=np.array([np.array(seq) for seq in train_seq], dtype=object),
                     test=np.array([np.array(seq) for seq in test_seq], dtype=object),
                     val=np.array([np.array(seq) for seq in val_seq], dtype=object), )


        print(f'Processed interim Data : saved to data/processed/data_seq.npz')


    """The code below is used to set up customized c device on computer"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print("You are using GPU acceleration.")
        print("Device name: ", torch.cuda.get_device_name(0))
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
    print("Number of cores: ", os.cpu_count())


    npzf = np.load(f'data/processed/data_seq.npz', allow_pickle=True)

    train_data = npzf['train']
    test_data = npzf['test']
    val_data = npzf['val']

    SEQUENCE_LENGTH = train_data.shape[1]

    print(f"Sequence length: {SEQUENCE_LENGTH}")

    # FORWARD_SEQ_LEN = int(SEQUENCE_LENGTH / 2)
    # LOOKBACK_SEQ_LEN = int(SEQUENCE_LENGTH / 2)

    # Testing to see if the model can predict the next point given the previous 2 points
    LOOKBACK_SEQ_LEN = 2
    FORWARD_SEQ_LEN = 1

    # Changed lookahead to 1 (predict 1)
    config = Namespace(hid_dim=128, emb_dim=128, out_dim=0,
                       lr=0.0003, momentum=0.9, epochs=N_EPOCHS, batch=BATCH_SIZE, opt='Adam', generate_type=True,
                       read_model=False, seq_len=FORWARD_SEQ_LEN, eval_epoch=EVAL_EPOCHS, s_min=1e-3, b_max=20,
                       lookahead=1, alpha=0.1, z_dim=128, beta=1e-3, dropout=0, num_head=2,
                       nlayers=3, num_points=NUM_BACKGROUND_POINTS, infer_nstep=10000, infer_limit=13, clip=1.0,
                       constrain_b='sigmoid', sample=True, decoder_n_layer=3)


    """
    Prepare logger
    """
    logger = logging.getLogger('full_lookahead{}batch{}'.format(config.lookahead, config.batch))
    logger.setLevel(logging.DEBUG)
    hdlr = logging.FileHandler('full_lookahead{}batch{}.log'.format(config.lookahead, config.batch))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    # this is really dumb, need to do it more efficiently
    # train_data_obj = np.empty(shape=(train_data.shape[0],), dtype=object)
    # for i in range(train_data.shape[0]):
    #     train_data_obj[i] = np.array(train_data[i], dtype=np.float64)

    # test_data_obj = np.empty(shape=(test_data.shape[0],), dtype=object)
    # for i in range(test_data.shape[0]):
    #     test_data_obj[i] = np.array(test_data[i], dtype=np.float64)

    # # this is really dumb, need to learn how to do this properly
    # val_data_obj = np.empty(shape=(val_data.shape[0],), dtype=object)
    # for i in range(val_data.shape[0]):
    #     val_data_obj[i] = np.array(val_data[i], dtype=np.float64)

    train_data_obj = np.apply_along_axis(lambda x: np.array(x, dtype=np.float64), 1, train_data)
    test_data_obj = np.apply_along_axis(lambda x: np.array(x, dtype=np.float64), 1, test_data)
    val_data_obj = np.apply_along_axis(lambda x: np.array(x, dtype=np.float64), 1, val_data)

    #%%
    print('Building Dataloader')

    trainset = SlidingWindowWrapper(train_data_obj, lookback=LOOKBACK_SEQ_LEN, lookahead=1, normalized=True)
    valset   = SlidingWindowWrapper(val_data_obj,  lookback=LOOKBACK_SEQ_LEN, lookahead=1, normalized=True, min=trainset.min, max=trainset.max)
    testset  = SlidingWindowWrapper(test_data_obj,  lookback=LOOKBACK_SEQ_LEN, lookahead=1, normalized=True, min=trainset.min, max=trainset.max)

    train_loader = DataLoader(trainset, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(valset, batch_size=config.batch, shuffle=False)
    test_loader = DataLoader(testset, batch_size=config.batch, shuffle=False)


    scales = (trainset.max - trainset.min).cpu().numpy()
    biases = trainset.min.cpu().numpy()
    # print(scales)
    # print(biases)

    # torch.autograd.set_detect_anomaly(True)
    #%%
    from model import DeepSTPP
    model = DeepSTPP(config, device)

    if TRAIN_MODEL:
        best_model = train(model, train_loader, val_loader, config, logger, device)
        # if directory does not exist, create it
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(best_model.state_dict(), f'models/data_seq_0.mod')


    model.load_state_dict(torch.load(f'models/data_seq_0.mod'))

    # scales = (trainset.max - trainset.min).cpu().numpy()
    # biases = trainset.min.cpu().numpy()

    # x_min, y_min, t_min = trainset.min
    # x_max, y_max, t_max = trainset.max

    T_NSTEP = 100
    X_NSTEP = 100
    Y_NSTEP = 100


    lambs, x_range, y_range, t_range, his_s, his_t = calc_lamb(model, test_loader, config, device, scales, biases,
                                                               t_nstep=T_NSTEP, x_nstep=X_NSTEP, y_nstep=Y_NSTEP,
                                                               # xmax=x_max, ymax=y_max, xmin=x_min, ymin=y_min,
                                                               total_time=TOTAL_TIME, round_time=False)


    # transpose all matrices in lambs list otherwise x/y are flipped !!!!
    lambs = [np.transpose(lamb, (1, 0)) for lamb in lambs]

    from plotter import plot_lambst_static, plot_lambst_interactive

    plot_lambst_interactive(lambs, x_range, y_range, t_range, heatmap=True)

    # create video directory if it does not exist
    if not os.path.exists('video'):
        os.makedirs('video')
    print('creating video')
    plot_lambst_static(lambs, x_range, y_range, t_range, history=(his_s, his_t), decay=2,
                        scaler=None, fps=12, fn=f'video/data_seq.mp4')


    print('here')



# %%

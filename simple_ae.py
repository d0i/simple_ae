#!/usr/bin/env python

import sys
import argparse
import pdb
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chainer
from chainer import optimizers
import chainer.functions as F
import chainer.links as L



class SimpleAutoEncoder(chainer.Chain):
    "simple autoencoder: input window size: (100,)"
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__(
            e1=L.Linear(100, 40),
            e2=L.Linear(40, 2),
            d1=L.Linear(2, 40),
            d2=L.Linear(40, 100))
        return
    def encode(self, x):
        h = F.relu(self.e1(x))
        z = F.relu(self.e2(h))
        return z
    def decode(self, z):
        h = F.relu(self.d1(z))
        y = self.d2(h)
        return y
    def __call__(self, x):
        return self.decode(self.encode(x))
    def loss(self, x):
        y = self(x)
        return F.mean_squared_error(x, y)

def sliding_window(data, steplen=5, window_width=100, offset=0):
    "return index and data of sliding window"
    for i in range(offset, len(data) - window_width - offset, steplen):
        yield i, data[i:i+window_width][None]
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser("simple autoencoder regression")
    parser.add_argument("--column", "-c", default=1, type=int, help="column to use")
    parser.add_argument("--train_ratio", "-r", default=0.1, type=float, help="ratio of data for training (from beginning)")
    parser.add_argument("--step", "-s", default=3, type=int, help="step of sliding window during training")
    parser.add_argument("--epoch", "-e", default=3, type=int, help="number of epochs")
    parser.add_argument("--window_width", "-w", default=100, type=int, help="window width")
    parser.add_argument("npy_file", type=str, help="numpy file with shape (n, 2)")

    args = parser.parse_args()

    # (n, 2), [[packets, bytes], ....]
    data = np.load(args.npy_file).astype(np.float32)
    data = np.squeeze(data[:,args.column])

    data = data/np.max(data)
    
    model = SimpleAutoEncoder()
    opt = optimizers.Adam()
    opt.use_cleargrads()
    opt.setup(model)

    train_data = data[:int(len(data)*args.train_ratio)]
    for e in range(args.epoch):
        bar = tqdm(desc="learn({}/{})".format(e, args.epoch), total=int((len(train_data)-args.window_width)/args.step), ascii=True)
        for di, w in sliding_window(train_data, steplen=args.step, window_width=args.window_width, offset=e):
            opt.update(model.loss, w) # using bytes
            bar.update()
        bar.close()
        total_loss = 0
        c=0
        for di, w in sliding_window(train_data, steplen=1, window_width=args.window_width):
            l = model.loss(w)
            total_loss += l.data
            c+=1
        print('loss={}\n'.format(total_loss/c))

    loss = np.zeros((data.shape[0],))
    bar = tqdm(desc="reg.", total=len(data)-args.window_width, ascii=True)
    for di, w in sliding_window(data, steplen=1, window_width=args.window_width):
        l = model.loss(w)
        loss[di] = l.data
        bar.update()
    bar.close()
    print('test_loss={}\n'.format(np.sum(loss)/len(loss)))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(data, label="bytes")
    ax2.plot(loss, label="anomaly", color="red", alpha=0.6)
    plt.savefig("loss.png")

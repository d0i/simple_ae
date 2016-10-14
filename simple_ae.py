#!/usr/bin/env python

import sys
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

    # (n, 2), [[packets, bytes], ....]
    data = np.load("test_data/201511111400_5ms_10M.npy").astype(np.float32)
    data = np.squeeze(data[:,1]) # use bytes

    data = data/np.max(data)
    
    model = SimpleAutoEncoder()
    opt = optimizers.Adam()
    opt.use_cleargrads()
    opt.setup(model)

    train_ratio = 0.1
    window_width = 100
    steplen=3
    epoch=3

    train_data = data[:int(len(data)*train_ratio)]
    for e in range(epoch):
        bar = tqdm(desc="learn({}/{})".format(e, epoch), total=int((len(train_data)-window_width)/steplen), ascii=True)
        for di, w in sliding_window(train_data, steplen=steplen, window_width=window_width, offset=e):
            opt.update(model.loss, w) # using bytes
            bar.update()
        bar.close()
        total_loss = 0
        c=0
        for di, w in sliding_window(train_data, steplen=1, window_width=window_width):
            l = model.loss(w)
            total_loss += l.data
            c+=1
        print('loss={}\n'.format(total_loss/c))

    loss = np.zeros((data.shape[0],))
    bar = tqdm(desc="reg.", total=len(data)-window_width, ascii=True)
    for di, w in sliding_window(data, steplen=1, window_width=window_width):
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

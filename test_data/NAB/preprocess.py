#!/usr/bin/env python

"""
tenuki filter
python preprocess.py foo.csv bar.npy
"""

import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plotfile", type=str, default=None)
    parser.add_argument("csvfile", type=str)
    parser.add_argument("npyfile", type=str)

    args = parser.parse_args()

    dataL = []
    with open(args.csvfile, "r") as f_csv:
        reader = csv.reader(f_csv)
        for row in reader:
            try:
                v = float(row[1])
                dataL.append([v])
            except ValueError:
                continue
    dataA = np.array(dataL)
    np.save(args.npyfile, dataA)

    if args.plotfile != None:
        plt.plot(dataA[:,0])
        plt.savefig(args.plotfile)

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import sys


def extract_from_csv(csv_file):
    fp = open(csv_file, 'r')
    print('opened', csv_file)
    nh = []
    acc = []
    for line in fp:
        line = line.strip().split(',')
        nh.append(int(line[0]))
        acc.append(float(line[1]))
    return nh, acc


def plot_accuracy(csv_files, out_name):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of human decisions')
    ax1.set_ylabel('Number of correct clusters')

    # combined_pairs = []
    for fn in csv_files:
        nh, acc = extract_from_csv(fn)
        _, tail = os.path.split(fn)
        prefix = tail.split('.')[0]
        ax1.plot(nh, acc, label=prefix)
    ax1.legend()

    if out_name is None:
        plt.show()
    else:
        plt.savefig(out_name)
        print('Saved plot to %s' % out_name)
        plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: %s out_name csv1 [csv2 .... csvn]' % sys.argv[0])
        sys.exit()

    plot_accuracy(sys.argv[2:], sys.argv[1])

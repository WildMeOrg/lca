# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys


def extract_pairs_from_csv(pairs_file, human_index, accuracy_index):
    fp = open(pairs_file, 'r')
    fp.readline()
    pairs = []
    for line in fp:
        line = line.strip().split(',')
        nh = int(line[human_index])
        acc = float(line[accuracy_index])
        pairs.append([nh, acc])
    return pairs


def densify_pairs(pairs, max_human):
    prev_nh = 0
    prev_acc = pairs[0][1]
    dense_prs = []
    for next_nh, next_acc in pairs:
        delta_acc = next_acc - prev_acc
        delta_nh = next_nh - prev_nh
        for nh in range(prev_nh, next_nh):
            acc = prev_acc + (nh - prev_nh) / delta_nh * delta_acc
            dense_prs.append([nh, acc])
        prev_nh, prev_acc = next_nh, next_acc
    dense_prs.append((prev_nh, prev_acc))

    last_nh, last_acc = dense_prs[-1]
    if last_nh < max_human:
        dense_prs.extend([[nh, last_acc] for nh in range(last_nh + 1, max_human + 1)])
    elif last_nh > max_human:
        while dense_prs[-1][0] > max_human:
            dense_prs.pop()

    return dense_prs


def plot_accuracy(csv_files, nh_index, acc_index, max_human, out_name):
    combined_pairs = []
    for fn in csv_files:
        new_pairs = extract_pairs_from_csv(fn, nh_index, acc_index)
        new_pairs = densify_pairs(new_pairs, max_human)
        if len(combined_pairs) == 0:
            combined_pairs = new_pairs
        else:
            for i in range(len(new_pairs)):
                combined_pairs[i][1] += new_pairs[i][1]

    nl = len(csv_files)
    human_decisions = [pr[0] for pr in combined_pairs]
    accuracy = [pr[1] / nl for pr in combined_pairs]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Number of human decisions')
    ax1.set_ylabel('Number of correct clusters')

    color = 'blue'
    ax1.plot(human_decisions, accuracy, color=color)

    if out_name is None:
        plt.show()
    else:
        plt.savefig(out_name)
        print('Saved plot to %s' % out_name)
        plt.close()

    return human_decisions, accuracy


def test_densify():
    pairs = [(1, 5.5), (2, 7.2), (5, 10.0), (6, 7.5), (10, 11.2)]
    print(densify_pairs(pairs, 15))
    print()
    print(densify_pairs(pairs, 10))
    print()
    pairs = [(0, 6), (4, 10), (7, 1), (8, 2), (9, 4), (10, 5), (10, 5), (11, 8)]
    print(densify_pairs(pairs, 9))


if __name__ == '__main__':
    # test_densify()

    if len(sys.argv) < 4:
        print('Usage: %s max_human out_prefix csv1 [csv2 ... csvn]' % sys.argv[0])
        sys.exit()

    nh_index = 0
    acc_index = 3
    max_human = int(sys.argv[1])
    out_name = sys.argv[2] + '.pdf'
    csv_files = sys.argv[3:]
    human_decisions, accuracy = plot_accuracy(
        csv_files, nh_index, acc_index, max_human, out_name
    )
    out_csv = sys.argv[2] + '.csv'
    fp = open(out_csv, 'w')
    for h, a in zip(human_decisions, accuracy):
        fp.write('%d, %1.3f\n' % (h, a))
    fp.close()
    print('Wrote summary to', out_csv)

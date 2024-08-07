import matplotlib.pyplot as plt
import json
from run import run as run_lca
from run_baseline import run as run_baseline
from tools import *
from init_logger import init_logger

species = 'giraffe'
x = 'num human'
y='frac correct'
# x = 'prob_human_error'
# y='error_rate'
config_path = 'configs/config_giraffe.yaml'


def plot(data_lca, data_baseline, x, y, save_path, xlabel=x, ylabel=y):

    x_values_lca = [entry[x] for entry in data_lca]
    y_values_lca = [entry[y] for entry in data_lca]

    x_values_baseline = [entry[x] for entry in data_baseline]
    y_values_baseline = [entry[y] for entry in data_baseline]


    sorted_data_lca = sorted(zip(x_values_lca, y_values_lca))
    sorted_data_baseline = sorted(zip(x_values_baseline, y_values_baseline))


    x_values_lca, y_values_lca = zip(*sorted_data_lca)
    x_values_baseline, y_values_baseline = zip(*sorted_data_baseline)


    plt.plot(x_values_lca, y_values_lca, marker='o', label='LCA')
    plt.plot(x_values_baseline, y_values_baseline, marker='o', label='Baseline')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(species)
    plt.grid(True)
    plt.legend()


    plt.savefig(save_path)


    plt.show()


def plot_one(species, config_path, x, y):


    results_lca = []
    results_baseline = []
    config = get_config(config_path)

        
    results_lca = run_lca(config)
    results_baseline = run_baseline(config)
    

    plot(results_lca, results_baseline, x, y, f'tmp/plots/{species}_1_{y}.eps', xlabel="Number of human reviews", ylabel='`%` of correct clusters')


def plot_two(species, config_path, x, y):

    human_review_accuracies = [0.80, 0.85, 0.90, 0.95, 0.98]
    results_lca = []
    results_baseline = []
    config = get_config(config_path)

    for human_review_accuracy in human_review_accuracies:
        config['lca']['prob_human_correct'] = human_review_accuracy
        result_lca = run_lca(config)[-1]
        result_lca['prob_human_error'] = 1-human_review_accuracy
        results_lca.append(result_lca)

        result_baseline = run_baseline(config)[0]
        result_baseline['prob_human_error'] = 1-human_review_accuracy
        results_baseline.append(result_baseline)

    

    plot(results_lca, results_baseline, x, y, f'tmp/plots/{species}_2.eps', xlabel="Probability of human review error", ylabel='Error rate')


def plot_per_cluster_size(species, config_path):


    results_lca = []
    config = get_config(config_path)

        
    results_lca = run_lca(config)[-1]

    print(results_lca)
    

    precision = [entry[0] for entry in results_lca['per size'].values()]
    cluster_sizes = [entry for entry in results_lca['per size'].keys()]

    recall = [entry[1] for entry in results_lca['per size'].values()]


    plt.plot(precision, cluster_sizes, marker='o', label='Precision')
    plt.plot(recall, cluster_sizes, marker='o', label='Recall')


    plt.xlabel('Cluster size')
    plt.ylabel('Accuracy')
    plt.title(species)
    plt.grid(True)
    plt.legend()


    plt.savefig('/home/kate/code/lca/lca/tmp/plots/giraffe_per_cluster_size.eps')
    plt.show()

init_logger()
# plot_one(species, config_path, x, y)
# plot_two(species, config_path, x, y)
# plot_per_cluster_size(species, config_path)

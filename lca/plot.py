import matplotlib.pyplot as plt
import json
from run import run as run_lca
from run_baseline import run as run_baseline
from run_baseline_topk import run_baseline_topk 
from tools import *
from init_logger import init_logger
import argparse

species = 'Beluga'
x = 'num human'
y='precision'
# x = 'prob_human_error'
# y='error_rate'
config_path = 'configs/config_beluga.yaml'




def plot_metrics(probs_human_correct, all_results_lca, all_results_baseline_topk, all_results_baseline_threshold, config, x, y, xlabel=x, ylabel=y, species='default', reachable=False):
    colors = plt.get_cmap('Set2')
    
    color_lca = colors(0)            # Color for LCA
    color_baseline_topk = colors(1)   # Color for Baseline Top-k
    color_baseline_threshold = colors(2)

    plt.figure()
    
    max_x = 5000

    i = 0
    for (prob_human_correct, results_lca, results_baseline_topk, results_baseline_threshold) in zip(probs_human_correct, all_results_lca, all_results_baseline_topk, all_results_baseline_threshold):
        num_annots = len(results_lca[2])
        random_color = colors(random.random())
        if reachable:
            results_lca =  results_lca[1]
        else:
            results_lca =  results_lca[0]
        results_baseline_topk = results_baseline_topk[0]
        results_baseline_threshold = results_baseline_threshold[0]
        num_individuals = results_lca[0]["num true clusters"]
        x_values_lca = [entry[x] for entry in results_lca]
        y_values_lca = [entry[y] for entry in results_lca]

        x_values_baseline_topk = [entry[x] for entry in results_baseline_topk]
        y_values_baseline_topk = [entry[y] for entry in results_baseline_topk]


        x_values_baseline_threshold = [entry[x] for entry in results_baseline_threshold]
        y_values_baseline_threshold = [entry[y] for entry in results_baseline_threshold]

        sorted_data_lca = sorted(zip(x_values_lca, y_values_lca))
        sorted_data_baseline_topk = sorted(zip(x_values_baseline_topk, y_values_baseline_topk))
        sorted_data_baseline_threshold = sorted(zip(x_values_baseline_threshold, y_values_baseline_threshold))

        x_values_lca, y_values_lca = zip(*sorted_data_lca)
        x_values_baseline_topk, y_values_baseline_topk = zip(*sorted_data_baseline_topk)
        x_values_baseline_threshold, y_values_baseline_threshold = zip(*sorted_data_baseline_threshold)
        max_x = max(max_x, x_values_lca[-1])

        if i==0:
            plt.plot(x_values_lca, y_values_lca, marker='o',  label=f'LCA', color=color_lca)
        else:
            plt.plot(x_values_lca, y_values_lca, marker='o', color=color_lca)

        
        plt.text(x_values_lca[-1], y_values_lca[-1], f'LCA {prob_human_correct}', fontsize=8)


        if i==0:
            plt.plot(x_values_baseline_threshold, y_values_baseline_threshold, marker='x', color=color_baseline_threshold, label=f'Baseline Threshold')
        else:
            plt.plot(x_values_baseline_threshold, y_values_baseline_threshold, marker='x', color=color_baseline_threshold)
        plt.text(x_values_baseline_threshold[3], y_values_baseline_threshold[3], f'Thr {prob_human_correct}', fontsize=8)

        if i==0:
            plt.plot(x_values_baseline_topk, y_values_baseline_topk, marker='*',  label=f'Baseline Top-k', color=color_baseline_topk)
        else:
            plt.plot(x_values_baseline_topk, y_values_baseline_topk, marker='*', color=color_baseline_topk)
        # plt.text(x_values_baseline_topk[-1], y_values_baseline_topk[-1], f'Top-k {prob_human_correct}', fontsize=8)

        

        i+=1

    plt.vlines([num_annots], 0, 1)
    plt.vlines([num_individuals], 0, 1)

    plt.text(num_annots, -0.05, f'Num Annots', ha='center', va='top', fontsize=7, color='gray')
    plt.text(num_individuals, -0.05, f'Num Inds: ', ha='center', va='top', fontsize=7, color='gray')



    plt.xlim(right=max_x)
    plt.ylim(0, 1.0) 

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(species)
    plt.grid(True)
    plt.legend()

    if reachable:
        filename = f'visualisations/baselines_vs_lca_{config["species"]}_{y}_reachable'
    else:
        filename = f'visualisations/baselines_vs_lca_{config["species"]}_{y}'


    plt.savefig(f'{filename}.eps')
    plt.savefig(f'{filename}.png')
    plt.close()
    # plt.show()

def plot_clusters(probs_human_correct, all_results_lca, all_results_baseline_topk, all_results_baseline_threshold, config, ylabel="Clusters", species='default'):
    colors = plt.get_cmap('Set2')
    x = 'num human'
    xlabel = "Number of human reviews"
    plt.figure()
    color_lca = colors(0)            # Color for LCA
    color_baseline_topk = colors(1)   # Color for Baseline Top-k
    color_baseline_threshold = colors(2)
   
    for (prob_human_correct, results_lca, results_baseline_topk, results_baseline_threshold) in zip(probs_human_correct, all_results_lca, all_results_baseline_topk, all_results_baseline_threshold):
            
        if (prob_human_correct != 0.98):
            continue
        reachable_lca = results_lca[1]
        # reachable_baseline_topk = results_baseline_topk[1]
        # reachable_baseline_threshold = results_baseline_threshold[1]
        results_lca =  results_lca[0]
        results_baseline_topk = results_baseline_topk[0]
        results_baseline_threshold = results_baseline_threshold[0]

        y_values_gt = [entry["num true clusters"] for entry in results_lca]
        y_values_reachable_gt = [entry["num true clusters"] for entry in reachable_lca]

        x_values_lca = [entry[x] for entry in results_lca]
        total_values_lca = [entry["num clusters"] for entry in results_lca]
        correct_values_lca = [entry["frac correct"] * entry["num true clusters"] for entry in results_lca]

        x_values_baseline_topk = [entry[x] for entry in results_baseline_topk]
        total_values_baseline_topk = [entry["num clusters"] for entry in results_baseline_topk]
        correct_values_baseline_topk = [entry["frac correct"] * entry["num true clusters"] for entry in results_baseline_topk]

        x_values_baseline_threshold = [entry[x] for entry in results_baseline_threshold]
        total_values_baseline_threshold = [entry["num clusters"] for entry in results_baseline_threshold]
        correct_values_baseline_threshold = [entry["frac correct"] * entry["num true clusters"] for entry in results_baseline_threshold]

        sorted_data_lca = sorted(zip(x_values_lca, total_values_lca, correct_values_lca, y_values_gt, y_values_reachable_gt))
        sorted_data_baseline_topk = sorted(zip(x_values_baseline_topk, total_values_baseline_topk, correct_values_baseline_topk))
        sorted_data_baseline_threshold = sorted(zip(x_values_baseline_threshold, total_values_baseline_threshold, correct_values_baseline_threshold))

        x_values_lca, total_values_lca, correct_values_lca, y_values_gt, y_values_reachable_gt = zip(*sorted_data_lca)
        x_values_baseline_topk, total_values_baseline_topk, correct_values_baseline_topk = zip(*sorted_data_baseline_topk)
        x_values_baseline_threshold, total_values_baseline_threshold, correct_values_baseline_threshold = zip(*sorted_data_baseline_threshold)

        plt.hlines([y_values_gt[0]], 0, 5000, label='True num clusters', color=colors(3))
        # plt.hlines([y_values_reachable_gt[0]], 0, 5000, label='Reachable num clusters')
        plt.plot(x_values_lca, y_values_reachable_gt, label='Reachable num clusters')
        
        plt.plot(x_values_lca, total_values_lca, label=f'LCA total {prob_human_correct}', color=colors(4))
        plt.plot(x_values_lca, correct_values_lca, label=f'LCA correct {prob_human_correct}', color=color_lca)

        plt.scatter(x_values_baseline_topk, total_values_baseline_topk, marker='*', label=f'Baseline Top-k total', color=colors(5))
        plt.scatter(x_values_baseline_topk, correct_values_baseline_topk, marker='*', label=f'Baseline Top-k correct', color=color_baseline_topk)

        plt.plot(x_values_baseline_threshold, total_values_baseline_threshold, label=f'Baseline threshold total ', color=colors(6))
        plt.plot(x_values_baseline_threshold, correct_values_baseline_threshold, label=f'Baseline threshold LCA correct {prob_human_correct}', color=color_baseline_threshold)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(species)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(right=5000)


    plt.savefig(f'visualisations/clusters_{config["species"]}_{y}.eps', bbox_inches='tight')
    plt.savefig(f'visualisations/clusters_{config["species"]}_{y}.png', bbox_inches='tight')
    plt.close()
    # plt.show()

def plot(data_lca, data_baseline, x, y, save_path, xlabel=x, ylabel=y):

    x_values_lca = [entry[x] for entry in data_lca]
    y_values_lca = [entry[y] for entry in data_lca]

    x_values_baseline = [entry[x] for entry in data_baseline]
    y_values_baseline = [entry[y] for entry in data_baseline]


    sorted_data_lca = sorted(zip(x_values_lca, y_values_lca))
    sorted_data_baseline = sorted(zip(x_values_baseline, y_values_baseline))


    x_values_lca, y_values_lca = zip(*sorted_data_lca)
    x_values_baseline, y_values_baseline = zip(*sorted_data_baseline)

    plt.figure()


    plt.plot(x_values_lca, y_values_lca, marker='o', label='LCA')
    plt.plot(x_values_baseline, y_values_baseline, marker='o', label='Baseline')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(species)
    plt.grid(True)
    plt.legend()


    plt.savefig(save_path)
    plt.close()

    # plt.show()


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
    plt.close()
    # plt.show()

# init_logger()
# plot_one(species, config_path, x, y)
# plot_two(species, config_path, x, y)
# plot_per_cluster_size(species, config_path)


def plot_top(config_path, x, y,  xlabel="Number of human reviews", ylabel='`%` of correct clusters', species='default'):


    results_lca = []
    results_baseline_threshold = []
    results_baseline_topk = [] 

    human_correct_probs = [1.0, 0.98, 0.96]
    config = get_config(config_path)
    exp_name = config['exp_name']

    for prob_human_correct in human_correct_probs:
        init_logger()
        config['exp_name'] = f"{exp_name}_{prob_human_correct}"
        config['lca']['edge_weights']['prob_human_correct'] = prob_human_correct

        results_baseline_topk.append(run_baseline_topk(config))
        init_logger()
        results_baseline_threshold.append(run_baseline(config))
        init_logger()
        results_lca.append(run_lca(config))
        

        save_pickle((human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold), f"/ekaterina/work/src/lca/lca/tmp/plot/{species}__{exp_name}.pickle")
    

    plot_metrics(human_correct_probs,results_lca, results_baseline_topk, results_baseline_threshold, config, x=x, y=y, xlabel=xlabel, ylabel=ylabel, species=species)
    # plot_clusters(human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold, config, species=species)


def get_topk(config):
    
    results_baseline_topk = [] 

    human_correct_probs = [1.0, 0.98, 0.96]
    config = get_config(config_path)
    exp_name = config['exp_name']

    for prob_human_correct in human_correct_probs:
        init_logger()
        config['exp_name'] = f"{exp_name}_{prob_human_correct}"
        config['lca']['edge_weights']['prob_human_correct'] = prob_human_correct

        results_baseline_topk.append(run_baseline_topk(config))
        init_logger()
    return results_baseline_topk

def plot_from_pickle(config_path, x, y,  xlabel="Number of human reviews", ylabel='`%` of correct clusters', species='default', reachable=False):
    
    config = get_config(config_path)
    exp_name = config['exp_name']

    
    human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold = load_pickle(f"/ekaterina/work/src/lca/lca/tmp/plot/{species}__{exp_name}.pickle")
    # results_baseline_topk = get_topk(config)
    plot_metrics(human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold, config, x=x, y=y, xlabel=xlabel, ylabel=ylabel, species=species, reachable=reachable)
    plot_clusters(human_correct_probs, results_lca, results_baseline_topk, results_baseline_threshold, config, species=species)


configs_with_species = [
    ('./configs/config_grevyszebra.yaml', 'Grevy\'s Zebra'),
    ('./configs/config_plainszebra.yaml', 'Plains Zebra'),
    ('./configs/config_forestelephants.yaml', 'Forest Elephants'),
    ('./configs/config_whaleshark.yaml', 'Whale Shark'),  # Uncomment if needed
    ('./configs/config_giraffe.yaml', 'Giraffe'),
    ('./configs/config_beluga.yaml', 'Beluga Whale')
    # ('./configs/config_spermwhale.yaml', 'Spermwhale'),
]



def run(config_path, species):
    plot_top(config_path, x='num human', y='precision', xlabel="Number of human reviews", ylabel='Precision', species=species)
    plot_from_pickle(config_path, x='num human', y='precision',  xlabel="Number of human reviews", ylabel='`Precision', species=species)
    plot_from_pickle(config_path, x='num human', y='f1 score',  xlabel="Number of human reviews", ylabel='`F1 score', species=species)
    plot_from_pickle(config_path, x='num human', y='recall',  xlabel="Number of human reviews", ylabel='`Recall', species=species)
    plot_from_pickle(config_path, x='num human', y='frac correct',  xlabel="Number of human reviews", ylabel='% of correct clusters', species=species)

    plot_from_pickle(config_path, x='num human', y='precision',  xlabel="Number of human reviews", ylabel='`Precision', species=species, reachable=True)
    plot_from_pickle(config_path, x='num human', y='f1 score',  xlabel="Number of human reviews", ylabel='`F1 score', species=species, reachable=True)
    plot_from_pickle(config_path, x='num human', y='recall',  xlabel="Number of human reviews", ylabel='`Recall', species=species, reachable=True)
    plot_from_pickle(config_path, x='num human', y='frac correct',  xlabel="Number of human reviews", ylabel='% of correct clusters', species=species, reachable=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to the YAML configuration file. Default: configs/default_config.yaml'
    )
    parser.add_argument(
        '--species',
        type=str,
        default='None',
        help='species name'
    )
    return parser.parse_args()


if __name__ == '__main__':
    init_logger()
    args = parse_args()
    config_path = args.config
    species = args.species


    run(config_path, species)

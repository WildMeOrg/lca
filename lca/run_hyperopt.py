import matplotlib.pyplot as plt
import json
from run import run as run_lca
from run_baseline import run as run_baseline
from run_baseline_topk import run_baseline_topk 
from tools import *
from init_logger import init_logger
import optuna
import datetime

def get_objective(config_path):
    init_logger()
    config = get_config(config_path)
    config['exp_name'] = config['exp_name'] + "_hyperopt"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"tmp/logs/{config['exp_name']}_{timestamp}.log"
    config['lca']['logging']['log_file'] = log_file_name
    config['lca']['logging']["update_log_file"] = False
    config['lca']['logging']["file_mode"] = "a"

    def objective(trial):

        config['lca']['distance_power'] = trial.suggest_categorical("distance_power", [0.5, 1, 2]) 
        config['lca']['iterations']['min_delta_converge_multiplier'] = trial.suggest_float('min_delta_converge_multiplier', 0.8, 1, step=0.01) # 0.95 
        config['lca']['iterations']['min_delta_stability_ratio'] = trial.suggest_int('min_delta_stability_ratio', 2, 12) # 4
        config['lca']['iterations']['num_per_augmentation'] = trial.suggest_int('num_per_augmentation', 1, 4) # 2
        config['lca']['iterations']['tries_before_edge_done'] = trial.suggest_int('tries_before_edge_done', 1, 8) # 4
        config['lca']['iterations']['ga_iterations_before_return'] = trial.suggest_int('ga_iterations_before_return', 4, 14) # 10
        config['lca']['iterations']['ga_max_num_waiting'] = trial.suggest_int('ga_max_num_waiting', 25, 100) # 50
 
        gt_results, _, _ = run_lca(config)
        result = gt_results[-1]['f1 score']

        # Or we can optimize this to get the best possible result while minimizing the number of human reviews necessary:
        # result = gt_results[-1]['f1 score']/(1 + gt_results[-1]['num human'])

        return result
    return objective, config


def run_hyperopt():
    config_path = './configs/config_plainszebra.yaml'

    objective, config = get_objective(config_path)

    logger = logging.getLogger('optuna')
    handler = logging.FileHandler(config['lca']["logging"]["log_file"], mode="a")
    handler.setLevel(config['lca']['logging']['log_level'])
    handler.setFormatter(get_formatter())
    logger.addHandler(handler)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    logger = logging.getLogger('lca')
    generate_ga_params(config['lca'])
    logger.info(f"Best trial: {study.best_trial}")



if __name__ == '__main__':
    run_hyperopt()


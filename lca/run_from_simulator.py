# -*- coding: utf-8 -*-
from datetime import datetime
import logging
import sys

import baseline
import cluster_tools as ct

import exp_scores as es
import graph_algorithm as ga
import simulator as sim
import weighter as wgtr


logger = logging.getLogger('lca')


def get_base_params():
    """
    Specify the parameters for the simulator
    """
    base_sim_params = dict()

    """
    The first set of parameters controls the formation of the nodes.
    The number of nodes in each cluster is generated from a gamma
    distribution. Since we require at least one node per cluster, the
    mean number of nodes in the clusters is 1 plus the mean of the
    distribution, and the mode number of nodes in the cluster is 1
    plus the mode of the distribution. The gamma distribution is
    controlled by the shape and the scale values:
          mean is shape*scale,
          mode is (shape-1)*scale
          variance is shape*scale**2 = mean*scale
          std dev = shape**0.5 * scale
    So when these are both 2 the mean is 4, the mode is 2
    and the variance is 4 (st dev 2).  And, when shape = 1,
    the mode is at 0 and we have an exponential distribution
    with the beta parameter of that distribution = scale.
    """
    base_sim_params['gamma_shape'] = 1
    base_sim_params['gamma_scale'] = 2
    base_sim_params['num_clusters'] = 512

    """
    The next set of parameters controls the formation of edges.  The
    first two specify the probability that the simulation of the
    ranker will produce an edge between any two nodes in a cluster,
    and the second controls how many edges are returned by the
    ranker. The third, pos_error_frac, is the expected fraction of
    edges within a cluster that will have randomly-generated weights
    that are negative; the expected fractions of edges between
    clusters that will have positive weights will be approximately the
    same. Finally, we have the fraction of decisions made by humans
    that are expected to be correct.
    """
    base_sim_params['p_ranker_correct'] = 0.85
    base_sim_params['num_from_ranker'] = 8
    base_sim_params['pos_error_frac'] = 0.15
    base_sim_params['p_human_correct'] = 0.95

    """
    The final parameters control the duration of each simulation and
    the number of times each simulation runs.
    """
    base_sim_params['max_human_mult'] = 3
    base_sim_params['num_simulations'] = 10

    """
    The algorithm parameters seem to be a bit less important here...
    """
    base_ga_params = {}
    base_ga_params['prob_human_correct'] = base_sim_params['p_human_correct']
    base_ga_params['min_delta_converge_multiplier'] = 2.0
    base_ga_params['max_human_decisions'] = (
        base_sim_params['max_human_mult'] * base_sim_params['num_clusters']
    )
    base_ga_params['min_delta_stability_ratio'] = 8
    base_ga_params['augmentation_names'] = ['vamp', 'human']
    base_ga_params['num_per_augmentation'] = 2
    base_ga_params['tries_before_edge_done'] = 4
    base_ga_params['ga_iterations_before_return'] = 100000  # convergence
    base_ga_params['ga_max_num_waiting'] = 50
    base_ga_params['should_densify'] = False
    base_ga_params['densify_min_edges'] = 5 * 4 / 2  # C(5, 2)
    base_ga_params['densify_frac'] = 0.5
    base_ga_params['log_level'] = logging.INFO
    base_ga_params['draw_iterations'] = False
    base_ga_params['drawing_prefix'] = 'drawing_lca'

    return base_sim_params, base_ga_params


def vary_gamma():
    base_sim, base_ga = get_base_params()
    sim_triples = []

    # Mean = 2, mode = 1, std_dev = 1
    sim_params = base_sim.copy()
    sim_params['gamma_shape'] = 1
    sim_params['gamma_scale'] = 1
    fname = 'mode_1_mean_2'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    # Mean = 3, mode = 1, std dev = 2
    sim_params['gamma_shape'] = 1
    sim_params['gamma_scale'] = 2
    fname = 'mode_1_mean_3'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    # Mean = 4, mode = 1, std dev = 3
    sim_params['gamma_shape'] = 1
    sim_params['gamma_scale'] = 3
    fname = 'mode_1_mean_4'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    # Mean = 3, mode = 2,
    sim_params['gamma_shape'] = 2
    sim_params['gamma_scale'] = 1
    fname = 'mode_2_mean_3'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    # Mean = 4, mode = 2
    sim_params['gamma_shape'] = 1.5
    sim_params['gamma_scale'] = 2
    fname = 'mode_2_mean_4'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    return sim_triples


def vary_human():
    base_sim, base_ga = get_base_params()
    sim_triples = []

    sim_params = base_sim.copy()
    sim_params['prob_human_correct'] = 0.90
    fname = 'human_p90'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['prob_human_correct'] = 0.92
    fname = 'human_p92'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['prob_human_correct'] = 0.94
    fname = 'human_p94'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['prob_human_correct'] = 0.96
    fname = 'human_p96'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['prob_human_correct'] = 0.98
    fname = 'human_p98'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    return sim_triples


def vary_verifier():
    base_sim, base_ga = get_base_params()
    sim_triples = []

    sim_params = base_sim.copy()
    sim_params['pos_error_frac'] = 0.05
    fname = 'verify_p05'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['pos_error_frac'] = 0.10
    fname = 'verify_p10'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['pos_error_frac'] = 0.15
    fname = 'verify_p15'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['pos_error_frac'] = 0.20
    fname = 'verify_p20'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['pos_error_frac'] = 0.25
    fname = 'verify_p25'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    return sim_triples


def vary_ranker():
    base_sim, base_ga = get_base_params()
    sim_triples = []

    sim_params = base_sim.copy()
    sim_params['p_ranker_corr'] = 0.70
    fname = 'ranker_p70'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['p_ranker_corr'] = 0.75
    fname = 'ranker_p75'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['p_ranker_corr'] = 0.80
    fname = 'ranker_p80'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['p_ranker_corr'] = 0.85
    fname = 'ranker_p85'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    sim_params = base_sim.copy()
    sim_params['p_ranker_corr'] = 0.90
    fname = 'ranker_p90'
    sim_triples.append([sim_params.copy(), base_ga.copy(), fname])

    return sim_triples


def one_simulation(out_path, file_prefix, sim_params, ga_params):
    import os

    print('==============================')
    print('Simulation name', file_prefix)
    path = os.path.join(out_path, file_prefix)
    if not os.path.exists(path):
        os.makedirs(path)
    path_and_file = os.path.join(path, file_prefix)

    log_file = path_and_file + '.log'
    # Delete log file if it exists
    try:
        os.remove(log_file)
    except Exception:
        pass

    """Configure the log file. This is repeated in the __init__ function
    for the graph_algorithm class, something that is only done here
    simulation information into the log file. It should not be done
    when running with "live" data.
    """
    import formatter

    handler = logging.FileHandler(log_file, mode='w')
    handler.setLevel(ga_params['log_level'])
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    log_format = '%(levelname)-6s [%(filename)18s:%(lineno)3d] %(message)s'
    logging.basicConfig(
        filename=log_file, level=ga_params['log_level'], format=log_format
    )

    logging.info('Simulation parameters')
    for k, v in sim_params.items():
        logging.info('    %a: %a' % (k, v))

    """
    Build the exponential weight generator. This
    """
    np_ratio = sim.find_np_ratio(
        sim_params['gamma_shape'],
        sim_params['gamma_scale'],
        sim_params['num_from_ranker'],
        sim_params['p_ranker_correct'],
    )

    logging.info('Negative / positive prob ratio %1.3f' % np_ratio)

    scorer = es.exp_scores.create_from_error_frac(sim_params['pos_error_frac'], np_ratio)
    wgtr_i = wgtr.weighter(scorer, human_prob=sim_params['p_human_correct'])

    """  Set convergence parameters """
    min_converge = -ga_params['min_delta_converge_multiplier'] * (
        wgtr_i.human_wgt(True) - wgtr_i.human_wgt(False)
    )
    ga_params['min_delta_score_converge'] = min_converge
    ga_params['min_delta_score_stability'] = (
        min_converge / ga_params['min_delta_stability_ratio']
    )

    for i in range(sim_params['num_simulations']):
        """Get the graph algorithm parameters"""
        logger.info('===================================')
        logger.info('Starting simulation %d' % i)
        print('Starting simulation', i)
        t0 = datetime.now()

    for i in range(sim_params['num_simulations']):
        """Get the graph algorithm parameters"""
        logger.info('===================================')
        logger.info('Starting simulation %d' % i)
        print('Starting simulation', i)
        t0 = datetime.now()

        """
        Build the simulator
        """
        # seed = 9314
        sim_i = sim.simulator(sim_params, wgtr_i)  # , seed=seed)
        init_edges, aug_names = sim_i.generate()
        init_clusters = []

        gai = ga.graph_algorithm(
            init_edges,
            init_clusters,
            aug_names,
            ga_params,
            sim_i.augmentation_request,
            sim_i.augmentation_result,
        )

        gai.set_trace_compare_to_gt_cb(
            sim_i.trace_start_human, sim_i.trace_iter_compare_to_gt
        )

        max_iterations = int(1e5)
        should_pause = converged = False
        iter_num = 0
        while iter_num < max_iterations and not converged:
            should_pause, iter_num, converged = gai.run_main_loop(iter_num)

        logger.info('')
        logger.info('Compare to ground truth')
        logger.info('By GT cluster length:')
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.gt_clustering)
        pct, pr, rec = ct.percent_and_PR(
            gai.clustering, gai.node2cid, sim_i.gt_clustering, sim_i.gt_node2cid
        )
        logger.info('Pct equal %.3f, Precision %.3f, Recall %.3f' % (pct, pr, rec))

        logger.info('')
        logger.info('Compare to reachable ground truth')
        logger.info('By reachable cluster length:')
        ct.compare_by_lengths(gai.clustering, gai.node2cid, sim_i.r_clustering)
        pct, pr, rec = ct.percent_and_PR(
            gai.clustering, gai.node2cid, sim_i.r_clustering, sim_i.r_node2cid
        )
        logger.info('Pct equal %.3f, Precision %.3f, Recall %.3f' % (pct, pr, rec))

        file_prefix = path_and_file + ('_%02d' % i)
        sim_i.csv_output(file_prefix + '_gt.csv', sim_i.gt_results)
        sim_i.csv_output(file_prefix + '_r.csv', sim_i.r_results)
        sim_i.generate_plots(file_prefix)

        b = baseline.baseline(sim_i)
        max_human_baseline = 10 * sim_params['num_clusters']

        b.all_iterations(0, max_human_baseline, 5)
        b.generate_plots(file_prefix + '_base')

        t1 = datetime.now()
        print('Simulation %d took %s' % (i, str(t1 - t0)))
        logging.info('Simulation %d took %s' % (i, str(t1 - t0)))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(
            'Usage: %s out_path [name1 .. namek]\n'
            'Where namei indicates the simulation experiment to run'
        )
        sys.exit()
    out_path = sys.argv[1]

    sim_triples = []
    if 'gamma' in sys.argv:
        sim_triples.extend(vary_gamma())

    if 'human' in sys.argv:
        sim_triples.extend(vary_human())

    if 'verifier' in sys.argv:
        sim_triples.extend(vary_verifier())

    if 'ranker' in sys.argv:
        sim_triples.extend(vary_ranker())

    for sim_params, ga_params, file_prefix in sim_triples:
        one_simulation(out_path, file_prefix, sim_params, ga_params)

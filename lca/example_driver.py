# -*- coding: utf-8 -*-
import argparse
import configparser
import logging
import json
import sys

import ga_driver
import db_interface_sim
import edge_generator_sim


logger = logging.getLogger('lca')


"""
This is a top-level driver for the LCA graph algorithm, written mostly
to illustrate use of the graph algorithm through smallconda examples and
simulations. Of particular note there are two objects created here
that must be replaced by objects that are connected to "real"
information about animals:  the edge and id database, and the
generator for edges through calls to verifiers or calls to human
reviewers.

Three key files are needed here:

1. The configuration file.  See the config.ini example

2. The JSON file of recent verifier ground truth positive and negative
probability results. Note as a reminder that both the values of the
probabilities AND the relative fraction of positive and negative
ground truth samples are important here.

3. The request JSON file, which includes the simulated database, the
simulated edge generator and the actual query request.  See
request_example.json

Note that the first two will always be needed even if this is running
"for real", as will the actual query (with the request JSON).  So in
an non-simulation, only the database and edge generator object need to
be replaced.
"""


def form_database(request):
    """
    From the request JSON object extract the database if it is there.
    If not, return an empty database. The JSON includes edge quads
    (n0, n1, w, aug_name) and a clustering dictionary.
    """
    edge_quads = []
    clustering_dict = dict()

    if 'database' not in request:
        return edge_quads, clustering_dict

    req_db = request['database']
    if 'quads' in req_db:
        edge_quads = req_db['quads']
    if 'clustering' in req_db:
        clustering_dict = {str(cid): c for cid, c in req_db['clustering'].items()}

    db = db_interface_sim.db_interface_sim(edge_quads, clustering_dict)
    return db


def form_edge_generator(request, db, wgtr):
    """
    Form the edge generator object. Unlike the database, the generator
    must be there for the small example / simulator to run.
    """
    try:
        gen_dict = request['generator']
    except KeyError:
        logger.info('Information about the edge generator must be in the request.')
        sys.exit(1)

    # Get hand-specified results from the verifier that aren't in the
    # database yet. These are prob_quads of the form (n0, n1, prob,
    # aug_name).  The weighter will be used to turn the prob into a
    # weight.
    prob_quads = []
    if 'verifier' in gen_dict:
        prob_quads = gen_dict['verifier']

    # Get human decisions of the form (n0, n1, bool). These will be
    # returned as new edges when first requested
    human_triples = []
    if 'human' in gen_dict:
        human_triples = gen_dict['human']

    # Get the ground truth clusters - used to generate edges that
    # aren't listed explicitly
    gt_clusters = []
    if 'gt_clusters' in gen_dict:
        gt_clusters = gen_dict['gt_clusters']

    # Get the nodes to be removed early in the computation.
    nodes_to_remove = []
    if 'nodes_to_remove' in gen_dict:
        nodes_to_remove = gen_dict['nodes_to_remove']

    # Get the number of steps between returning edge generation
    # results. If this value is 0 then they are returned immediately
    # upon request.
    delay_steps = 0
    if 'delay_steps' in gen_dict:
        delay_steps = gen_dict['delay_steps']

    edge_gen = edge_generator_sim.edge_generator_sim(
        db, wgtr, prob_quads, human_triples, gt_clusters, nodes_to_remove, delay_steps
    )
    return edge_gen


def extract_requests(request, db):
    try:
        req_dict = request['query']
    except KeyError:
        logger.info('Information about the GA query itself must be in the request JSON.')
        sys.exit(1)

    # 1. Get the verifier result quads (n0, n1, prob, aug_name).
    verifier_results = []
    if 'verifier' in req_dict:
        verifier_results = req_dict['verifier']

    # 2. Get the human decision result triples (n0, n1, bool)
    # No error checking is used
    human_decisions = []
    if 'human' in req_dict:
        human_decisions = req_dict['human']

    # 3. Get the list of existing cluster ids to check
    cluster_ids_to_check = []
    if 'cluster_ids' in req_dict:
        cluster_ids_to_check = req_dict['cluster_ids']

    for cid in cluster_ids_to_check:
        logger.info(cid)
        logger.info(cluster_ids_to_check)
        if not db.cluster_exists(cid):
            logger.info('GA request cluster id %s does not exist' % cid)
            raise ValueError('Cluster id does not exist')

    return verifier_results, human_decisions, cluster_ids_to_check


from init_logger import init_logger
if __name__ == '__main__':
    print("START")
    init_logger()

    logger = logging.getLogger('lca')
    print(logger.handlers)
    parser = argparse.ArgumentParser('example_driver.py')
    parser.add_argument(
        '--ga_config', type=str, required=True, help='graph algorithm config INI file'
    )
    parser.add_argument(
        '--verifier_gt',
        type=str,
        required=True,
        help='json file containing verification algorithm ground truth',
    )
    parser.add_argument(
        '--request',
        type=str,
        required=True,
        help='json file continain graph algorithm request info',
    )
    parser.add_argument(
        '--db_result', type=str, help='file to write resulting json database'
    )

    # 1. Configuration
    args = parser.parse_args()
    config_ini = configparser.ConfigParser()
    config_ini.read(args.ga_config)

    print('step 1')

    # 2. Recent results from verification ground truth tests. Used to
    # establish the weighter.
    with open(args.verifier_gt, 'r') as fn:
        verifier_gt = json.loads(fn.read())
    print('step 2')

    # 3. Form the parameters dictionary and weight objects (one per
    # verification algorithm).
    ga_params, wgtrs = ga_driver.params_and_weighters(config_ini, verifier_gt)
    if len(wgtrs) > 1:
        logger.info('Not currently handling more than one weighter!!')
        sys.exit(1)
    wgtr = wgtrs[0]

    print('step 3')

    # 4. Get the request dictionary, which includes the database, the
    # actual request edges and clusters, and the edge generator edges
    # and ground truth (for simulation).
    with open(args.request, 'r') as fn:
        request = json.loads(fn.read())

    db = form_database(request)
    edge_gen = form_edge_generator(request, db, wgtr)
    verifier_req, human_req, cluster_req = extract_requests(request, db)

    print('step 4')

    # 5. Form the graph algorithm driver
    driver = ga_driver.ga_driver(
        verifier_req, human_req, cluster_req, db, edge_gen, ga_params
    )

    print('step 5')

    # 6. Run it. Changes are logged.
    ccPIC_gen = driver.run_all_ccPICs()
    changes_to_review = list(ccPIC_gen)
    logger.info(changes_to_review)

    print('step 6')

    # 7. Commit changes. Record them in the database and the log
    # file.
    # TBD

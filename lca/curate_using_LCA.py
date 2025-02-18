import db_interface
import edge_generator
import ga_driver
import os
import json
import csv
import pandas as pd
import random
import logging
from weighter import weighter
from tools import *
from ga_driver import IterationHalt, IterationPause, IterationConverged
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np


"""
The underlying database of edges in the multigraph.
For simplicity this is just a csv with one row per
edge giving the node/annotation ids, the weight and
the augmentation name. It is subclassed from
db_interface class which interfaces to the LCA code.
"""
def create_file(path):
    dir_name = os.path.dirname(path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

    return path

def create_json_file(path):
    dir_name = os.path.dirname(path)
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    data = {}
    with open(path, 'w', newline='') as f:
        json.dump(data, f)

    return path





class db_interface_generic(db_interface.db_interface):
    def __init__(self, db_file, clustering_file, clustering):
        self.db_file = db_file if os.path.exists(db_file) else create_file(db_file)
        self.clustering_file = clustering_file if os.path.exists(clustering_file) else create_json_file(clustering_file)

        
        quads = []
        with open(self.db_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                n0, n1, weight, method = int(row[0]), int(row[1]), int(row[2]), row[3]
                quads.append([n0, n1, weight, method])

        # Output stats to the logger
        super(db_interface_generic, self).__init__(quads, clustering, are_edges_new=False)



    def add_edges_db(self, quads):

        with open(self.db_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(quads)


    def commit_cluster_change_db(self, clustering):
        
        write_json(clustering, self.clustering_file)
        return clustering

        
    
       

    # Pretty sure this is not needed.  Eliminate from db_interface.py as well.
    def edges_from_attributes_db(self, n0, n1):
        quads = []
        attrib = self.edge_graph[n0][n1]
        for a, wgt in attrib.items():
            if a == 'human':
                quads.extend([(n0, n1, w, 'human') for w in wgt])
            else:
                quads.append((n0, n1, wgt, a))
        return quads
    

    """
    Pretty sure this is currently not needed
    def commmit_cluster_change_db(self, cc):
        raise NotImplementedError()
    """

AUG_NAME_HUMAN = 'human'


def is_aug_name_algo(aug_name):
    return aug_name != AUG_NAME_HUMAN


def is_aug_name_human(aug_name):
    return aug_name == AUG_NAME_HUMAN


class edge_generator_generic(edge_generator.edge_generator):  # NOQA
    def __init__(self, db, wgtr, verifiers):
        self.verifiers = verifiers
        super(edge_generator_generic, self).__init__(db, wgtr)

    def get_edge_requests(self):
        return self.edge_requests

    def set_edge_requests(self, new_edge_requests):
        """
        Assign the edge requests. Note that this is a replacement, not an append.
        """
        self.edge_requests = new_edge_requests
        return self.edge_requests

    def edge_request_cb_async(self):
        """
        Called from LCA through super class method to handle augmentation requests.
        Requests for verification probability edges are handled immediately and saved
        as results for LCA to grab. Requests for human reviews are saved for sending
        to the web interface
        """
        verifier_quads = []  # Requests for verifier
        human_review_requests = []  # Requests for human review
        for edge in self.get_edge_requests():
            n0, n1, aug_name = edge
            if is_aug_name_algo(aug_name):
                verifier_quads.append((n0, n1, self.verifiers[aug_name]([(n0, n1)]), aug_name))
            else:
                human_review_requests.append(edge)

        
        # Convert the quads to edge weight quads; while doing so, add
        # to the database.
        wgt_quads = self.new_edges_from_verifier(verifier_quads)

        # Set the verifier results for LCA to pick up.
        self.edge_results += wgt_quads

        # Set the human review requests to be sent to the web interface.
        self.set_edge_requests(human_review_requests)

    def ingest_human_reviews(self, review_triples, quit=False):
        self.quit = quit
        new_edge_results = self.new_edges_from_human(review_triples)
        self.edge_results += new_edge_results
        requests_to_keep = {
            (n0, n1) : req for n0, n1, req in self.get_edge_requests()
        }

        for n0, n1, _ in review_triples:
            pr = (n0, n1)
            if pr in requests_to_keep and \
                is_aug_name_human(requests_to_keep[pr]):
                del requests_to_keep[pr]

        requests_to_keep = [
            (n0, n1, requests_to_keep[(n0, n1)])
            for (n0, n1) in requests_to_keep
        ]

        self.set_edge_requests(requests_to_keep)





def find_uncertainty_ranges(all_vals):
    # Extract the scores and labels (True/False)
    X = np.array([[score] for (_, _, score, label) in all_vals])  # Using score as the feature
    y = np.array([label for (_, _, score, label) in all_vals]).astype(int)  # True/False as 1/0
    
    # Standardize the score for better logistic regression performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Logistic Regression with probability estimates (Bayesian models provide this inherently)
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_scaled, y)
    
    # Predict probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of being True (1)
    
    # Identify regions where the probabilities are near 0.5 (mixed True/False)
    uncertainty_threshold = 0.2  # Define how close to 0.5 the probability should be
    uncertain_indices = np.where((probabilities > 0.5 - uncertainty_threshold) &
                                 (probabilities < 0.5 + uncertainty_threshold))[0]
    
    uncertain_vals = [all_vals[i][2] for i in uncertain_indices]
    
    # for i in uncertain_indices:
    #     print(f"Score: {all_vals[i][2]}, Probability of True: {probabilities[i]}, Uncertain: True")
    
    return [np.min(uncertain_vals), np.max(uncertain_vals)]


"""
Given 
(1) a list of verifier edge triples (n0, n1, score), where
    n0 < n1 are nodes and score is the verifier score, 
(2) a method to call to obtain human reviews on a list of 
    (n0, n1) pairs
(3), (4) target number positive and negative human decisions

Return:
(1) a list of positively-reviewed edge triples
(2) a list of negatively-reviewed edge triples
(3) a boolean signal from the human reviewer to quit LCA

In its usual behavior the function will return enough
positive and negative triples and will not return a quit 
signal.  
"""
def generate_wgtr_calibration_ground_truth(verifier_edges,
                                           human_reviewer,
                                           num_pos_needed,
                                           num_neg_needed,
                                           num_bins=1):

    # Method: The range of scores is divided up into equally-sized
    # bins and the verifier edges are assigned to the bins based
    # on their score.  After shuffling each bins, edges are picked
    # one by one from each bin and saved in a list until the bins
    # are empty. Edges from the list are sent to the human_reviewer
    # in small batches and the results are saved.

    # 1. For the bins
    # num_bins = 10
    scores = [s for _, _, s in verifier_edges]
    min_score = min(scores)
    max_score = max(scores)
    delta_score = (max_score - min_score) / (num_bins-1)
    bins = [[] for _ in range(num_bins)]
    for n0, n1, s in verifier_edges:
        i = int((s - min_score) / delta_score)
        bins[i].append((n0, n1, s))


    # 2. Shuffle each bin
    for i in range(num_bins):
        random.shuffle(bins[i])

    # 3. Pull edges from the bins
    edge_nodes = []
    edge_scores = {}
    i = 0
    while len(bins) > 0:
        if len(bins[i]) == 0:
            del bins[i]
            if i == len(bins):
                i = 0
        else:
            n0, n1, s = bins[i][-1]
            edge_nodes.append((n0, n1))
            edge_scores[(n0, n1)] = s
            del bins[i][-1]
            i = (i + 1) % len(bins)

    # 4. Send to the reviewer in small batches.
    num_in_batch = max(4, (num_pos_needed + num_neg_needed) // 2)
    pos_triples = []
    neg_triples = []
    i = 0
    quit_lca = False
    while i < len(edge_nodes):
        j = min(i + num_in_batch, len(edge_nodes))
        reviews, quit_lca = human_reviewer(edge_nodes[i: j])
        # if quit_lca:
        #     break
        for n0, n1, b in reviews:
            e = (n0, n1, edge_scores[(n0, n1)])
            if b:
                pos_triples.append(e)
            else:
                neg_triples.append(e)
        if len(pos_triples) >= num_pos_needed and len(neg_triples) >= num_neg_needed:
            break
        i = j


    # all_vals = [(n0, n1, s, True) for triple in pos_triples]
    # all_vals += [(n0, n1, s, False) for triple in neg_triples]
    # r = 0.2
    # N = 30
    # all_vals = [(n0, n1, s, True) for n0, n1, s in pos_triples]
    # all_vals += [(n0, n1, s, False) for n0, n1, s in neg_triples]
    # uncertain_vals = find_uncertainty_ranges(all_vals)
    # print(uncertain_vals)

    # uncertain_edges = [(n0, n1, s) for n0, n1, s in verifier_edges if (uncertain_vals[0]) <= s <= (uncertain_vals[1])]
    
    # if len(uncertain_edges) < N:
    #     print(f"Only {len(uncertain_edges)} edges found in the uncertain range.")
    #     selected_edges = uncertain_edges
    # else:
    #     selected_edges = random.sample(uncertain_edges, N)
    
    # # Send selected edges to human reviewer
    # reviews, quit_lca = human_reviewer([(n0, n1) for n0, n1, s in selected_edges])
    

    # for n0, n1, b in reviews:
    #     e = (n0, n1, edge_scores[(n0, n1)])
    #     if b:
    #         pos_triples.append(e)
    #     else:
    #         neg_triples.append(e)

    return pos_triples, neg_triples, quit_lca
    # return pos_triples, neg_triples, quit_lca, None, [None]



def generate_ground_truth_random(verifier_edges,
                                           human_reviewer,
                                           num_pos_needed,
                                           num_neg_needed):

    verifier_edges = list(verifier_edges)
    # Shuffle verifier edges randomly
    random.shuffle(verifier_edges)

    # Prepare lists to store results
    pos_triples = []
    neg_triples = []

    # Send edges to the human reviewer in small batches
    num_in_batch = max(4, (num_pos_needed + num_neg_needed) // 2)
    i = 0
    quit_lca = False

    while i < len(verifier_edges):
        # Select a batch of edges
        j = min(i + num_in_batch, len(verifier_edges))
        edges_batch = verifier_edges[i:j]

        # Get reviews from the human reviewer
        reviews, quit_lca = human_reviewer([(n0, n1) for n0, n1, s in edges_batch])

        for n0, n1, b in reviews:
            e = (n0, n1, next(s for x0, x1, s in verifier_edges if x0 == n0 and x1 == n1))
            if b:
                pos_triples.append(e)
            else:
                neg_triples.append(e)

        # Break if enough positive and negative triples are found
        if len(pos_triples) >= num_pos_needed and len(neg_triples) >= num_neg_needed:
            break

        i = j

    return pos_triples, neg_triples, quit_lca


def generate_ground_truth_full_dataset(verifier_edges,
                                           human_reviewer):

    verifier_edges = list(verifier_edges)
    # Shuffle verifier edges randomly
    random.shuffle(verifier_edges)

    # Prepare lists to store results
    pos_triples = []
    neg_triples = []

    # Send edges to the human reviewer in small batches
    i = 0
    quit_lca = False
    num_in_batch = 100
    while i < len(verifier_edges):
        # Select a batch of edges
        j = min(i + num_in_batch, len(verifier_edges))
        edges_batch = verifier_edges[i:j]

        # Get reviews from the human reviewer
        reviews, quit_lca = human_reviewer([(n0, n1) for n0, n1, s in edges_batch])

        for n0, n1, b in reviews:
            e = (n0, n1, next(s for x0, x1, s in verifier_edges if x0 == n0 and x1 == n1))
            if b:
                pos_triples.append(e)
            else:
                neg_triples.append(e)

        i = j

    return pos_triples, neg_triples, quit_lca


def generate_calib_weights(pos, neg, get_score):

    pos_result = []
    neg_result = []

    for n0, n1, _ in pos:
        e = (n0, n1, get_score(n0, n1))
        pos_result.append(e)

    for n0, n1, _ in neg:
        e = (n0, n1, get_score(n0, n1))
        neg_result.append(e)
    
    return pos_result, neg_result

def generate_wgtr_calibration_random_bins(verifier_edges,
                                           human_reviewer,
                                           needed_total,
                                           min_samples_from_bin,
                                           num_bins=2):

    # Method: The range of scores is divided up into equally-sized
    # bins and the verifier edges are assigned to the bins based
    # on their score.  After shuffling each bins, edges are picked
    # one by one from each bin and saved in a list until the bins
    # are empty. Edges from the list are sent to the human_reviewer
    # in small batches and the results are saved.

    # 1. For the bins
    # num_bins = 10
    scores = [s for _, _, s in verifier_edges]
    min_score = min(scores)
    max_score = max(scores)
    delta_score = (max_score - min_score) / (num_bins-1)
    bins = [[] for _ in range(num_bins)]
    for n0, n1, s in verifier_edges:
        i = int((s - min_score) / delta_score)
        bins[i].append((n0, n1, s))

    logger = logging.getLogger('lca')
    logger.info(f"Bin sizes: {[len(b) for b in bins]}")

    pos_triples = []
    neg_triples = []
    # 3. Pull edges from the bins
    for i in range(len(bins)):
        batch_sz = np.clip(int(needed_total * len(bins[i])/len(scores)), min_samples_from_bin, len(bins[i]))
        rnd_edges = [bins[i][j] for j in np.random.choice(len(bins[i]), batch_sz, replace=False)]
        edge_nodes = [(n0, n1) for (n0, n1, s) in rnd_edges]
        edge_scores = {(n0, n1):s for (n0, n1, s) in rnd_edges}
        reviews, quit_lca = human_reviewer(edge_nodes)
        # if quit_lca:
        #     break
        for n0, n1, b in reviews:
            e = (n0, n1, edge_scores[(n0, n1)])
            if b:
                pos_triples.append(e)
            else:
                neg_triples.append(e)
    return pos_triples, neg_triples, quit_lca

"""
verifier_alg: function:
    Argument(s): 
    . List of annotation node pairs (n0, n1)
    Returns:
    . List of non-negative, bounded matching scores, one for
      each node pair

verifier_name: string
    Gives the name of the verifier algorithm in use.

human_reviewer: function:
    Function to ask human reviewer to decide if pairs of
    of annotations show the same individual or different
    individuals. Eventually this will be extended to allow
    reviewers to indicate that a reliable decision cannot
    be made.
    Argument(s):
    . List of annotation node pairs (n0, n1)
    Returns:
    . List of review triples in the form (n0, n1, bool)
    . Boolean flag, which if set to be True indicates
      that LCA should stop     

wgts_calib_dict:
    Dictionary containing verification scores/probs and associated
    human decisions. The keys of the dictionary are the verification
    (augmentation) algorithm name, and the values are dictionaries of
    probs for pairs marked positive (same animal) and negative.  Note that
    the relative proportion of positive and negative scores can matter.
    This is entirely used for the weighting calibration.
            ALGO_AUG_NAME: {
                'gt_positive_probs': new_pos_probs,
                'gt_negative_probs': new_neg_probs,
            }

edge_db_file: string
    Path to database file of edges. Each edge quad is represented
    by a row in this csv file.

current_clustering: dictionary
    The current clustering, represented as a mapping from cluster
    id to list (or set) of annotation/node ids.  Could be empty.

========================
For the curate function:
........................
verifier_results: 
    list of 3-tuples (a1, a2, score) where a1 < a2 are the annotations
    and score is the output of the verifier.  These typically come
    from applying verification to the results of ranking.
    Importantly, (a1, a2) should not already have a verification
    edge in the database

human_reviews:
    list of 3-tuples (a1, a2, decision) where a1 < a2 are the
    annotations and decision is a binary result of human review. Unlike
    the verifier_results, (a1, a2) is allowed to already have a review

cluster_ids_to_check:
    list of cluster ids to check.  Usually empty (at least for now).
"""
class curate_using_LCA(object):
    def __init__(self,
                 verifier_algs,
                 verifier_names,
                 human_reviewer,
                 wgtrs_calib_dict,
                 edge_db_file,
                 clustering_file, #maybe comes from db file
                 current_clustering,
                 lca_config):
        self.verifier_algs = verifier_algs
        self.verifier_names = verifier_names
        self.human_reviewer = human_reviewer
        self.wgtrs_calib_dict = wgtrs_calib_dict
        self.lca_config = lca_config
        self.edge_db_file = edge_db_file
        self.current_clustering = current_clustering
        self.clustering_file = clustering_file

        # 1. Create weighter from calibration

        self.wgtrs = ga_driver.generate_weighters(
            self.lca_config, self.wgtrs_calib_dict
        )
        
        # 2. Update delta score thresholds in the lca config
        # This should probably be in a LCA-proper file
        multiplier = self.lca_config['min_delta_converge_multiplier']
        ratio = self.lca_config['min_delta_stability_ratio']
        human_gt_positive_weight = weighter.human_wgt(is_marked_correct=True)
        human_gt_negative_weight = weighter.human_wgt(is_marked_correct=False)
        human_gt_delta_weight = human_gt_positive_weight - human_gt_negative_weight
        convergence = -1.0 * multiplier * human_gt_delta_weight
        stability = convergence / ratio
        self.lca_config['min_delta_score_converge'] = convergence
        self.lca_config['min_delta_score_stability'] = stability

        # 3.
        self.db = db_interface_generic(self.edge_db_file, self.clustering_file, self.current_clustering)

        # 4. Create the edge generators
        self.edge_gen = edge_generator_generic(self.db, self.wgtrs, self.verifier_algs)
    
    def save_active_clusters(self, active_clusters, clustering):
        autosave_file = self.lca_config['autosave_file']
        output =  {
            'cluster_ids_to_check': set([cid for val in active_clusters.values() for cid in val ]),
            'clustering': clustering

        } 
        write_json(output, autosave_file)


    def curate(self,
               verifier_results,      # [(a1, a2, score)]
               human_reviews,        # [(a1, a2, decision)]
               cluster_ids_to_check = []):

        #  Add the name of the verifier to each ranker / verifier result
        #  This is not necessary for the human reviews.

        logger = logging.getLogger('lca')
        # verifier_results = [
        #     (n0, n1, s, self.verifier_name)
        #     for n0, n1, s in verifier_results
        # ]

        #  Create the graph algorithm driver. This requires that at least one
        #  of verifier_results, human_reviews and cluster_ids_to_check be 
        #  non-empty.  The main outcome of the intializer is the creation of
        #  a list of subgraphs (ccPICs) that could be impacted by the potential
        #  changes in the graph.
        driver = ga_driver.ga_driver(
            verifier_results,
            human_reviews,
            cluster_ids_to_check,
            self.db,
            self.edge_gen,
            self.lca_config
        )

        #  Create the iterator to run through each ccPIC subgraph.  The iterator
        #  yields either (a) StopIteration when it is done, (b) nothing, or 
        #  (c) a list of cluster changes.  See the while loop for more information.
        ga_gen = driver.run_all_ccPICs(
            yield_on_paused=True,
        )

        cluster_changes = []
    
        while True:
            try:
                #  Run the next step. This step stops when LCA is completely done,
                #  or when enough human review requests have built up that LCA
                #  needs to wait for them to be completedr when a ccPIC is finished 
                #  and has yielded changes that must be recorded before the next
                #  ccPIC starts.
                #  
                #  (a) If the yield is a StopIteration exception then the LCA
                #  clustering work is done and we break out of the while loop
                next_cluster_changes = next(ga_gen)
            except StopIteration:
                break
            
            
            if type(next_cluster_changes) is IterationPause:
                #  (b) If the change_to_review is None we are in the middle of LCA
                #  applied to a single CCPIC and more human reviews are needed. These
                #  review requests will have been previously communicated to the edge
                #  generator.
                requested_edges = self.edge_gen.get_edge_requests()
                requested_edges = [(n0, n1) for n0, n1, _ in requested_edges]

                logger.info(f'Received {len(requested_edges)} human review requests')


                
                self.db.commit_cluster_changes(next_cluster_changes.cluster_changes, temporary=True)
                # logger.info(f'Cluster changes {next_cluster_changes.cluster_changes}')
                self.save_active_clusters(driver.active_clusters, self.db.latest_clustering)

                # return (cluster_changes, False)

                #  Need to add the ability to stop the computation here....
                review_triples, quit = self.human_reviewer(requested_edges)
                
                self.edge_gen.ingest_human_reviews(review_triples, quit)
            else:
                #  4c. The third case for the end of an iteration (a yield from
                #      run_all_ccPICs) is the completion of a single ccPIC. In this
                #      case the cluster changes from the ccPIC are return for review
                #      and commitment.
                cluster_changes.append(next_cluster_changes.cluster_changes)
                self.db.commit_cluster_changes(next_cluster_changes.cluster_changes)

                
        return (cluster_changes, True)





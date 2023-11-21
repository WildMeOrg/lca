import db_interface
import edge_generator
import ga_driver


"""
The underlying database of edges in the multigraph.
For simplicity this is just a csv with one row per
edge giving the node/annotation ids, the weight and
the augmentation name. It is subclassed from
db_interface class which interfaces to the LCA code.
"""
class db_interface_generic(db_interface.db_interface):
    def __init__(self, db_file, clustering):
        self.db_file = db_file

        with open(self.db_file, 'r') as f:
            lines = f.read().split('\n')
        lines = [line.strip().split() for line in lines]
        quads = [
            (int(n0), int(n1), int(wgt), aug_name)    # need to make this more general -- UUIDs for example
            for n0, n1, wgt, aug_name in lines
        ]

        # Output stats to the logger
        super(db_interface_generic, self).__init__(quads, clustering, are_edges_new=False)

    def add_edges_db(self, quads):
        with open(self.db_file, 'a') as f:
            for n0, n1, w, aug_name in quads:
                f.write(f'{n0},{n1},{w},{aug_name}\n')
        """
        The following was for recording attributes in the NetworkX graph.
        It does not appear to be needed
        for n0, n1, w, aug_name in quads:
            attrib = self.edge_graph[n0][n1]
            if aug_name == 'human':
                if 'human' not in attrib:
                    attrib['human'] = []
                attrib['human'].append(w)
            else:
                attrib[aug_name] = w
        """

    """
    Pretty sure this is not needed.  Eliminate from db_interface.py as well.
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

    """
    Pretty sure this is currently not needed
    def commmit_cluster_change_db(self, cc):
        raise NotImplementedError()
    """


class edge_generator_generic(edge_generator.edge_generator):  # NOQA
    def __init__(self, db, wgtr, verifier_alg, verifier_name):
        self.verifier_alg = verifier_alg
        self.verifier_name = verifier_name
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
        requested_verifier_edges = []  # Requests for verifier
        human_review_requests = []  # Requests for human review
        for edge in self.get_edge_requests():
            n0, n1, aug_name = edge
            if is_aug_name_algo(aug_name):
                requested_verifier_edges.append((n0, n1))
            else:
                human_review_requests.append(edge)

        # Get edge scores (probabilities) from the verifier
        scores = self.verifier_alg(requested_verifier_edges)

        # Convert these scores/probabilities to quads
        verifier_quads = [
            (e[0], e[1], s, self.verifier_name)
            for e, s in zip(requested_verifier_edges, scores)
        ]

        # Convert the quads to edge weight quads; while doing so, add
        # to the database.
        wgt_quads = self.new_edges_from_verifier(verifier_quads)

        # Set the verifier results for LCA to pick up.
        self.edge_results += wgt_quads

        # Set the human review requests to be sent to the web interface.
        self.set_edge_requests(human_review_requests)

    def ingest_human_reviews(self, review_triples):
        new_edge_results = self.new_edges_from_human(review_triples)
        self.edge_results += new_edge_results
        requests_to_keep = {
            (n0, n1) : req for n0, n1, req in self.get_edge_requests()
        }

        for n0, n1, _ in review_triples:
            if (n0, n1) in requests_to_keep and \
                requests_to_keep[(n0, n1)] == 'human':
                del requests_to_keep[(n0, n1)]

        requests_to_keep = [
            (n0, n1, requests_to_keep[(n0, n1)])
            for (n0, n1) in requests_to_keep
        ]

        self.set_edge_requests(requests_to_keep)


"""
"""
def generate_wgtr_calibration_ground_truth(verifier_edges,
                                           human_reviewer,
                                           num_pos_needed,
                                           num_neg_needed)
{
    # Follow style in _plugin.py for sampling.

    # Return both the human review triples and the dictionary of 
    # scores for both positively and negatively reviewed pairs.

    sort(verifier_edges)
}



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
                 verifier_alg,
                 verifier_name,
                 human_reviewer,
                 wgtrs_calib_dict,
                 edge_db_file,
                 current_clustering,
                 lca_config):
        self.verifier_alg = verifier_alg
        self.verifer_name = verifier_name
        self.human_reviewer = human_reviewer
        self.wgtrs_calib_dict = wgtrs_calib_dict
        self.lca_config = lca_config

        # 1. Create weighter from calibration

        wgtrs = ga_driver.generate_weighters(
            self.lca_config, self.wgtrs_calib_dict
        )
        self.wgtr = wgtrs[0]    # In the future we can consider muliple algorithms

        # 2. Update delta score thresholds in the lca config
        # This should probably be in a LCA-proper file
        multiplier = actor.lca_config['min_delta_converge_multiplier']
        ratio = actor.lca_config['min_delta_stability_ratio']
        human_gt_positive_weight = actor.wgtr.human_wgt(is_marked_correct=True)
        human_gt_negative_weight = actor.wgtr.human_wgt(is_marked_correct=False)
        human_gt_delta_weight = human_gt_positive_weight - human_gt_negative_weight
        convergence = -1.0 * multiplier * human_gt_delta_weight
        stability = convergence / ratio
        self.lca_config['min_delta_score_converge'] = convergence
        self.lca_config['min_delta_score_stability'] = stability

        # 3.
        self.db = db_interface_generic(self.edge_db_file, self.current_clustering)

        # 4. Create the edge generators
        self.edge_gen = edge_generator_generic(self.db, self.wgtr, self.verifier_alg):


    def curate(self,
               verifier_results,      # [(a1, a2, score)]
               human_reviews,        # [(a1, a2, decision)]
               cluster_ids_to_check = []):

        #  Add the name of the verifier to each ranker / verifier result
        #  This is not necessary for the human reviews.
        verifier_results = [
            (n0, n1, s, self.verifier_name)
            for n0, n1, s in verifier_results
        ]

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
            self.edge_generator,
            self.lca_params
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
                next_cluster_changes = next(self.ga_gen)
            except StopIteration:
                break

            if next_cluster_changes is None:
                #  (b) If the change_to_review is None we are in the middle of LCA
                #  applied to a single CCPIC and more human reviews are needed. These
                #  review requests will have been previously communicated to the edge
                #  generator.
                requested_edges = self.edge_gen.get_edge_requests()
                requested_edges = [(n0, n1) for n0, n1, _ in requested_edges]

                logger.info(f'Received {len(requested_edges)} human review requests')

                #  Need to add the ability to stop the computation here....
                review_triples = self.human_reviewer(requested_edges)
                self.edge_gen.ingest_human_reviews(review_triples)
            else:
                #  4c. The third case for the end of an iteration (a yield from
                #      run_all_ccPICs) is the completion of a single ccPIC. In this
                #      case the cluster changes from the ccPIC are return for review
                #      and commitment.
                cluster_changes.append(next_cluster_changes)

        return cluster_changes

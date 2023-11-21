# -*- coding: utf-8 -*-
from wbia.control import controller_inject
from wbia import constants as const
from wbia.web.graph_server import GraphClient, GraphActor
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN

import numpy as np
import os
import logging
import utool as ut
from functools import partial

from wbia_lca import db_interface
from wbia_lca import edge_generator

# import configparser
import threading
import random

# import json

from wbia_lca import ga_driver

import tqdm

logger = logging.getLogger('wbia_lca')


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']

SIMREVIEW_IDENTITY = 'user:simreview'
SHORTCIRCUIT_REVIEW_IDENTITY = 'user:shortcircuit'

HUMAN_AUG_NAME = 'human'
HUMAN_IDENTITY = 'user:web'
HUMAN_IDENTITY_PREFIX = '%s:' % (HUMAN_IDENTITY.split(':')[0],)
ALGO_AUG_NAME = 'vamp'
ALGO_IDENTITY = 'algo:vamp'
ALGO_IDENTITY_PREFIX = '%s:' % (ALGO_IDENTITY.split(':')[0],)

HUMAN_CORRECT_RATE = 0.98

# LOG_DECISION_FILE = 'lca.decisions.csv'
LOG_LCA_FILE = 'lca.log'

OVERALL_CONFIG_FILE = '/data/db/lca_plugin_overall.cfg'

"""
Need to be careful of types and values. Review feedback coming from
the web interface is in the form of strings and values like 'match'
(the imported value POSTV) and 'nomatch' (the imported value NEGTV),
whereas review decisions are stored in the database as ints.  The
following constants are int.
"""
UNREVIEWED = const.EVIDENCE_DECISION.UNREVIEWED
NEGATIVE = const.EVIDENCE_DECISION.NEGATIVE
POSITIVE = const.EVIDENCE_DECISION.POSITIVE
INCOMPARABLE = const.EVIDENCE_DECISION.INCOMPARABLE
UNKNOWN = const.EVIDENCE_DECISION.UNKNOWN
DECISION_TYPES = [NEGATIVE, POSITIVE]  # eventually, INCOMPARABLE


"""
First we have convenience functions for interpretting what comes
back from the web UI and for converting back and forth between WBIA
naming conventions and LCA naming conventions.
"""


def is_aug_name_human(aug_name):
    return aug_name == HUMAN_AUG_NAME


def is_aug_name_algo(aug_name):
    return aug_name == ALGO_AUG_NAME


def is_identity_human(identity):
    if identity is None:
        return False
    return identity.startswith(HUMAN_IDENTITY_PREFIX)


def is_identity_algo(identity):
    if identity is None:
        return False
    return identity.startswith(ALGO_IDENTITY_PREFIX)


def convert_aug_name_to_identity(aug_name_list):
    identity_list = []
    for aug_name in aug_name_list:
        if is_aug_name_human(aug_name):
            identity = HUMAN_IDENTITY
        elif is_aug_name_algo(aug_name):
            identity = ALGO_IDENTITY
        else:
            raise ValueError()
        identity_list.append(identity)
    return identity_list


def convert_identity_to_aug_name(identity_list):
    aug_name_list = []
    for identity in identity_list:
        if is_identity_human(identity):
            aug_name = HUMAN_AUG_NAME
        elif is_identity_algo(identity):
            aug_name = ALGO_AUG_NAME
        else:
            raise ValueError()
        aug_name_list.append(aug_name)
    return aug_name_list


def convert_lca_cluster_id_to_wbia_name_id(lca_cluster_id):
    wbia_name_id = int(lca_cluster_id)
    return wbia_name_id


def convert_wbia_name_id_to_lca_cluster_id(wbia_name_id):
    lca_cluster_id = '%05d' % (wbia_name_id,)
    return lca_cluster_id


def convert_lca_node_id_to_wbia_annot_id(lca_node_id):
    wbia_annot_id = int(lca_node_id)
    return wbia_annot_id


def convert_wbia_annot_id_to_lca_node_id(wbia_annot_id):
    lca_node_id = '%05d' % (wbia_annot_id,)
    return lca_node_id


def order_nodes_in_edge(edge):
    if edge[0] > edge[1]:
        return (edge[1], edge[0])
    else:
        return edge


def should_short_circuit_review(ibs, edge):
    """
    Given an edge that is about to be reviewed by a human, are there
    time/location constraints on the two annotations that form the edge
    that tell us immediately that they can't be pictures of the same
    animal.

    AVI:  INSERT YOUR FUNCTION HERE

    """
    return False


"""
Removed:

GGR specifics;
. def get_dates(ibs, gid_list, gmt_offset=3.0):
. def get_ggr_stats(ibs, valid_aids, valid_nids):
  wrong version of
. def progress_db(actor, gai, iter_num):
"""


def progress_db(actor, gai, iter_num):
    # logger.info('progress_db, iter_num %d' % iter_num)
    pass


class db_interface_wbia(db_interface.db_interface):  # NOQA
    """
    Provide the interface between WBIA and LCA for edges and clusters.
    """

    def __init__(self, actor):
        """
        Pull out the current edge weights and clustering from WBIA
        and prepare them for LCA. Communicate them to LCA via the
        super class initializer
        """
        self.controller = actor
        self.infr = actor.infr
        self.ibs = actor.infr.ibs

        """
        CHECK ME: I believe this is no longer needed.
        """
        self.max_auto_reviews = 1
        self.max_human_reviews = 10
        self.max_reviews = self.max_auto_reviews + self.max_human_reviews

        # Get the current clustering based on the WBIA database of names
        clustering = self._get_existing_clustering()

        #  Get the current edge weights and form them as quads for construction
        #  of LCA's db.
        ibs = self.ibs
        aids = self.infr.aids
        edge_wgt_rowids = ibs.get_edge_weight_rowids_between(aids)
        prs = ibs.get_edge_weight_aid_tuple(edge_wgt_rowids)
        a1s = [convert_wbia_annot_id_to_lca_node_id(min(e)) for e in prs]
        a2s = [convert_wbia_annot_id_to_lca_node_id(max(e)) for e in prs]
        weights = ibs.get_edge_weight_value(edge_wgt_rowids)
        ids = ibs.get_edge_weight_identity(edge_wgt_rowids)
        quads = list(zip(a1s, a2s, weights, ids))
        num_human_edges = sum([1 for i in ids if i == HUMAN_IDENTITY])
        num_verify_edges = len(ids) - num_human_edges
        logger.info(
            f'Initialized database with {num_human_edges} human review'
            + f' edges and {num_verify_edges} verifier-weighted edges'
        )

        #  Initialize the super-class, telling it that the edges quads it
        #  received are not new --- they are already in the underlying
        #  database and therefore do not need to be added again.
        super(db_interface_wbia, self).__init__(quads, clustering, are_edges_new=False)

    def _get_existing_clustering(self):
        """
        Access the current clustering in the WBIA database and convert it
        to LCA's form.
        """
        aids = self.infr.aids
        nids = self.ibs.get_annot_nids(aids)
        clustering = dict()
        for aid, nid in zip(aids, nids):
            cluster_label_ = convert_wbia_name_id_to_lca_cluster_id(nid)
            cluster_node_ = convert_wbia_annot_id_to_lca_node_id(aid)
            if cluster_label_ not in clustering:
                clustering[cluster_label_] = []
            clustering[cluster_label_].append(cluster_node_)

        logger.info(
            f'At start, clustering has {len(clustering)} names. Here are the first few'
        )
        for nid in sorted(clustering.keys())[:25]:
            clustering[nid] = sorted(clustering[nid])
            logger.info(f'\tcluster NID {nid}: {clustering[nid]}')

        return clustering

    def _cleanup_edges(self, max_auto=None, max_human=None):
        """
        I'm pretty sure this procedure to remove edge weights
        that are beyond the maximum allowed is no longer needed.
        I have commented out its single call in the following function.
        """
        weight_rowid_list = self.ibs.get_edge_weight_rowids_between(self.infr.aids)
        if max_auto is None:
            max_auto = self.max_auto_reviews
        if max_human is None:
            max_human = self.max_human_reviews

        """
        CHECK ME: Ensure that this works properly
        """
        self.ibs.check_edge_weights(
            weight_rowid_list=weight_rowid_list,
            max_auto=max_auto,
            max_human=max_human,
        )

    def add_edges_db(self, quads):
        """
        Add edges in the form of LCA quads to the WBIA edge_weight database.
        This requires format conversion before the addition.
        """
        aid_1_list = list(
            map(convert_lca_node_id_to_wbia_annot_id, ut.take_column(quads, 0))
        )
        aid_2_list = list(
            map(convert_lca_node_id_to_wbia_annot_id, ut.take_column(quads, 1))
        )
        value_list = ut.take_column(quads, 2)
        aug_name_list = ut.take_column(quads, 3)
        identity_list = convert_aug_name_to_identity(aug_name_list)

        weight_rowid_list = self.ibs.add_edge_weight(
            aid_1_list, aid_2_list, value_list, identity_list
        )
        # self._cleanup_edges()
        return weight_rowid_list

    def edges_from_attributes_db(self, n0, n1):
        """
        Get all edge weights associated with the LCA node pair n0 and n1.
        This requires converting from LCA node ids to WBIA aids.
        """
        a1 = convert_lca_node_id_to_wbia_annot_id(n0)
        a2 = convert_lca_node_id_to_wbia_annot_id(n1)
        edges = [(a1, a2)]
        weight_rowid_list = self.ibs.get_edge_weight_rowids_from_edges(edges)
        weight_rowid_list = weight_rowid_list[0]
        weight_rowid_list = sorted(weight_rowid_list)

        node_0_list = [n0] * len(weight_rowid_list)
        node_1_list = [n1] * len(weight_rowid_list)
        value_list = self.ibs.get_edge_weight_value(weight_rowid_list)
        identity_list = self.ibs.get_edge_weight_identity(weight_rowid_list)
        aug_name_list = convert_identity_to_aug_name(identity_list)

        quads = list(zip(node_0_list, node_1_list, value_list, aug_name_list))

        return quads

    def commit_cluster_change_db(self, cc):
        """
        Commit a cluster change. Such a change is a minimal set of interdependent
        changes.  There are two particular jobs: (1) create a change dictionary
        with WBIA names, and (2) save clusters as WBIA names. The latter adds
        NIDs if they didn't exist before and removes NIDs that have changed more
        than just as the addition of query annotations. This will require REVISITING
        because it makes no effort to preserve metadata that is associated
        with names that have been removed.
        """
        logger.info(
            '[commit_cluster_change_db] Requested to commit cluster change: %r' % (cc,)
        )
        change = cc.serialize()

        change['type'] = change.pop('change_type').lower()

        change['added'] = sorted(
            list(
                map(
                    convert_lca_node_id_to_wbia_annot_id,
                    change.pop('query_nodes'),
                )
            )
        )

        change['removed'] = sorted(
            list(
                map(
                    convert_lca_node_id_to_wbia_annot_id,
                    change.pop('removed_nodes'),
                )
            )
        )
        change['old'] = []
        change['new'] = []

        logger.info(f"Change type {change['type']}")

        old_clustering = change.pop('old_clustering')
        nids_to_remove = []
        for old_lca_cluster_id in old_clustering:
            old_lca_node_ids = old_clustering[old_lca_cluster_id]
            wbia_aids = list(map(convert_lca_node_id_to_wbia_annot_id, old_lca_node_ids))
            change['old'].append(wbia_aids)
            old_nids = self.ibs.get_annot_nids(wbia_aids)
            old_nids = list(set([nid for nid in old_nids if nid > 0]))

            logger.info(f'Old name id {old_nids}, annotations {wbia_aids}')

            if len(old_nids) > 1:
                logger.info(
                    'Error: annots in this old cluster are spread across multiple names'
                )
                nids_to_remove.extend(old_nids)  # Set to remove these
            elif len(old_nids) == 1 and change['type'] in [
                'removed',
                'split',
                'merge',
                'merge/split',
            ]:
                nids_to_remove.extend(old_nids)

        if len(nids_to_remove) > 0:
            logger.info(f'Deleting old nids {nids_to_remove}')
            self.ibs.delete_names(nids_to_remove)

        new_clustering = change.pop('new_clustering')
        for new_lca_cluster_id in new_clustering:
            new_lca_node_ids = new_clustering[new_lca_cluster_id]
            wbia_aids = list(map(convert_lca_node_id_to_wbia_annot_id, new_lca_node_ids))
            change['new'].append(wbia_aids)
            prev_nids = self.ibs.get_annot_nids(wbia_aids)
            prev_nids = list(set([nid for nid in prev_nids if nid > 0]))

            logger.info(f'Previous nids {prev_nids} for cluster of aids {wbia_aids}')
            if len(prev_nids) == 0 or change['type'] in [
                'split',
                'merge',
                'merge/split',
            ]:
                self.ibs.set_annot_names_to_next_name(wbia_aids)
                new_nid = self.ibs.get_annot_nids(wbia_aids[0])
                logger.info(f'Created new name nid {new_nid} for aids {wbia_aids}')
            elif change['type'] == 'extension':
                assert len(prev_nids) == 1
                prev_aids = self.ibs.get_name_aids(prev_nids[0])
                new_aids = list(set(wbia_aids) - set(prev_aids))
                self.ibs.set_annot_name_rowids(new_aids, prev_nids * len(new_aids))
                logger.info(
                    f'Added aids {new_aids} to prev aids {prev_aids} for nid {prev_nids[0]}'
                )
            else:
                logger.info(f'Cluster {wbia_aids} with nid {prev_nids[0]} is unchanged')

        return change


class edge_generator_wbia(edge_generator.edge_generator):  # NOQA
    """
    Subclass of LCA edge_generator for WBIA-specific creation of edges
    Its most important job is to manage the request for augmentations,
    dividing this into verification requests that can handled immediately,
    and human review requests that must be queued up for review and then
    gathered (through the feedback method). The requests are kept internally
    in their LCA form
    """

    def __init__(self, db, wgtr, controller):
        """
        db and wgtr are LCA object
        The controller is needed for access to the pluging functionality.
        """
        self.controller = controller
        super(edge_generator_wbia, self).__init__(db, wgtr)

    def _cleanup_edges(self):
        """
        99.9% sure this isn't needed.
        """
        clean_edge_requests = []
        # logger.info(f'_cleanup_edges, edge_requests are {self.edge_requests}')
        for edge in self.edge_requests:
            n0, n1, aug_name = edge
            aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
            aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
            if aid1 > aid2:
                aid1, aid2 = aid2, aid1
            n0 = convert_wbia_annot_id_to_lca_node_id(aid1)
            n1 = convert_wbia_annot_id_to_lca_node_id(aid2)
            clean_edge = (n0, n1, aug_name)
            clean_edge_requests.append(clean_edge)
        self.edge_requests = clean_edge_requests

    def get_edge_requests(self):
        self._cleanup_edges()
        return self.edge_requests

    def set_edge_requests(self, new_edge_requests):
        """
        Assign the edge requests. Note that this is a replacement, not an append.
        """
        self.edge_requests = new_edge_requests
        self._cleanup_edges()
        return self.edge_requests

    def edge_request_cb_async(self):
        actor = self.controller
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
                aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
                aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
                requested_verifier_edges.append((aid1, aid2))
            else:
                human_review_requests.append(edge)

        # Get edge probabilities from the verifier
        probs = actor._candidate_edge_probs(requested_verifier_edges)

        # Convert these probabilities to quads
        verifier_quads = [
            (
                convert_wbia_annot_id_to_lca_node_id(e[0]),
                convert_wbia_annot_id_to_lca_node_id(e[1]),
                p,
                ALGO_AUG_NAME,
            )
            for e, p in zip(requested_verifier_edges, probs)
        ]

        # Convert the probability quads to edge weight quads; while doing so, add
        # to the database.
        wgt_quads = self.new_edges_from_verifier(verifier_quads)

        # Set the verifier results for LCA to pick up.
        self.edge_results += wgt_quads

        # Set the human review requests to be sent to the web interface.
        self.set_edge_requests(human_review_requests)

        logger.info(f'Added {len(wgt_quads)} verifier edge quads')
        logger.info(f'Kept {len(human_review_requests)} requests for human review')

    def add_feedback(self, edge, evidence_decision=None, **kwargs):
        """
        This function is called when human reviews are given for edges
        requested by LCA.
        """
        if evidence_decision is None:
            decision = UNREV
        else:
            decision = evidence_decision

        logger.info(f'db add_feedback: edge {edge}, decision {decision}')
        assert isinstance(decision, str)

        if decision == POSTV:
            flag = True
        elif decision == NEGTV:
            flag = False
        elif decision == INCMP:
            flag = None
        else:
            # UNREV, UNKWN
            return

        """
        Task 1: Provide the new edge to LCA, which will grab it as
        part of the next iteration.
        """
        aid1, aid2 = edge
        n0 = convert_wbia_annot_id_to_lca_node_id(aid1)
        n1 = convert_wbia_annot_id_to_lca_node_id(aid2)
        human_triples = [
            (n0, n1, flag),
        ]
        new_edge_results = self.new_edges_from_human(human_triples)
        self.edge_results += new_edge_results

        """
        Task 2:  Remove the edge request for this pair now that a result
        has been returned.
        """
        found_edge_requests = []
        keep_edge_requests = []
        for triple in self.get_edge_requests():
            n0_, n1_, aug_name = triple
            if is_aug_name_human(aug_name):
                if n0 == n0_ and n1 == n1_:
                    found_edge_requests.append(edge)
                    continue
            keep_edge_requests.append(triple)
        args = (
            len(found_edge_requests),
            len(keep_edge_requests),
        )
        logger.info(
            'Found %d human edge requests to remove, kept %d requests in queue' % args
        )
        self.set_edge_requests(keep_edge_requests)

        """
        Task 3: Remove any unknown / unreviewed reviews from the database for this edge
        """
        ibs = self.controller.infr.ibs
        logger.info(f'Still working on new edges {edge}')
        rowids = ut.flatten(ibs.get_review_rowids_from_edges([edge]))
        decisions = ibs.get_review_decision(rowids)
        logger.info(f'Decision list is {decisions}')
        is_unknown = [d not in DECISION_TYPES for d in decisions]
        to_remove = ut.compress(rowids, is_unknown)
        logger.info(f'Num unknown/unrev to remove {len(to_remove)}')
        if len(to_remove) > 0:
            ibs.delete_review(to_remove)


def filter_prob_outliers(probs):
    m = np.mean(probs)
    sd = np.std(probs)
    lower = m - 2 * sd  # should parameterize this
    high = m + 2 * sd
    probs = [p for p in probs if lower <= p <= high]
    return probs


class LCAActor(GraphActor):
    """
    The main plugin class.
    """

    def __init__(
        actor, *args, ranker='hotspotter', verifier='vamp', num_waiting=10, **kwargs
    ):
        """
        The init function is a bit of mess and needs more abstraction.
        """

        """
        The overall config is in a file in a fixed location, communicated
        via the global constant full path. Propagating here via the kwargs
        dictionary would be a better solution, but it would require a number
        of changes to WBIA that I'm currently unwilling to make.
        """
        actor.overall_config = ut.load_cPkl(OVERALL_CONFIG_FILE)

        # The inference object for the Hotspotter (LNBNN) and VAMP
        actor.infr = None
        actor.graph_uuid = None

        """
        Attribute variables for sending to LCA's ga_driver function
        """
        actor.db = None
        actor.edge_gen = None
        actor.driver = None
        actor.ga_gen = None
        actor.changes = None
        actor.new_verifier_quads = None  # Verifier prob edges that are new to LCA
        actor.new_human_review_triples = None  # Human review triples that are new to LCA

        # The following is marked True if the review probabilities come from a file
        # This should only be False after starting up a new collection of reviews
        # for calications
        actor.LCA_calib_reviews_are_from_file = False

        # Set to True after LCA's weighter has been calibrated
        actor.has_LCA_calib = False

        # Dict mapping WBIA candidate edges to verifier prob. Always computed
        actor.probs_for_review_edges = None

        # Edge pairs before adding to the web-based human review interface
        actor.before_review_for_LCA_calib = None

        # Edges pairs currently in the review interface
        actor.in_review_for_LCA_calib = None

        # Dictionary of verifier probabilities for human-review edges
        actor.reviews_for_LCA_calib = None

        # Set to True if tried to open pkl file of ground truth clusters and failed.
        actor.failed_to_open_gt_clusters_filepath = False

        # Ground truth mapping from WBIA aid to cluster id for simulation
        actor.gt_aid_clusters = None  #

        # Statistics about the effectiveness of short-circuiting. These are
        # output to the logging file
        actor.use_short_circuiting = actor.overall_config['use_short_circuiting']
        actor.short_circuit_count = 0
        actor.no_short_circuit_count = 0
        actor.short_circuit_correct = 0
        actor.short_circuit_incorrect = 0

        actor.resume_lock = threading.Lock()

        actor.phase = 0
        actor.loop_phase = 'init'

        #  Set the verifier config. Really only Grevy's for now.
        ranker = actor.overall_config['ranker']
        verifier = actor.overall_config['verifier']

        #  Set the ranker config file. This will require generalization.
        if ranker == 'hotspotter':
            actor.ranker_config = {}
        elif ranker == 'pie_v2':
            actor.ranker_config = {
                'pipeline_root': 'PieTwo',
                'use_knn': False,
            }
        else:
            raise ValueError('Unsupported Ranker')

        if verifier == 'vamp':
            actor.verifier_config = {
                'verifier': 'vamp',
            }
        elif verifier == 'pie_v2':
            actor.verifier_config = {
                'verifier': 'pie_v2',
            }
        else:
            raise ValueError('Unsupported Verifier')

        #  LCA's internal config.
        actor.lca_config = {
            'aug_names': [
                ALGO_AUG_NAME,
                HUMAN_AUG_NAME,
            ],
            'prob_human_correct': HUMAN_CORRECT_RATE,
            # DEFAULT
            # 'min_delta_converge_multiplier': 0.95,
            # 'min_delta_stability_ratio': 8,
            # 'num_per_augmentation': 2,
            # 'tries_before_edge_done': 4,
            # 'ga_max_num_waiting': 1000,
            # EXTENSIVE
            'min_delta_converge_multiplier': 0.98,  # was 1.5,
            'min_delta_stability_ratio': 4,
            'num_per_augmentation': 2,
            'tries_before_edge_done': 4,
            'ga_max_num_waiting': num_waiting,
            'ga_iterations_before_return': 100,  # IS THIS USED?
            'log_level': logging.INFO,
            'log_file': LOG_LCA_FILE,
            'draw_iterations': False,
            'drawing_prefix': 'wbia_lca',
        }

        prob_human_correct = actor.lca_config.get(
            'prob_human_correct', HUMAN_CORRECT_RATE
        )

        #  plugin config for controlling calibration and simulations.
        actor.config = {
            'LCA_calib_num_to_queue': 25,  # was 500
            'LCA_calib_required_reviews': 50,
            'LCA_calib_max_reviews': 1000,
            'simreview.enabled': True,
            'simreview.prob_human_correct': prob_human_correct,
        }

        from wbia_lca import formatter

        handler = logging.FileHandler(actor.lca_config['log_file'])
        handler.setLevel(actor.lca_config['log_level'])
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        super(LCAActor, actor).__init__(*args, **kwargs)

    def _init_infr(actor, aids, dbdir, **kwargs):
        """
        Initiallize the inference object. Most important step is
        initializing the verifier
        """
        import wbia

        # Moved these down here because they aren't often changed.
        actor.infr_config = {
            # 'manual.n_peek': 100,
            'inference.enabled': True,
            'ranking.enabled': True,
            'ranking.ntop': 10,
            'redun.enabled': True,
            'redun.enforce_neg': True,
            'redun.enforce_pos': True,
            'redun.neg.only_auto': False,
            'redun.neg': 2,
            'redun.pos': 2,
            'refresh.window': 20,
            'refresh.patience': 20,
            'refresh.thresh': np.exp(-2),
            'algo.hardcase': False,
        }

        assert dbdir is not None
        assert actor.infr is None
        ibs = wbia.opendb(dbdir=dbdir, use_cache=False, web=False, force_serial=True)

        # Create the reference AnnotInference
        logger.info('starting via actor with ibs = %r' % (ibs,))
        actor.infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)
        actor.infr.print('started via actor')
        actor.infr.print('config = {}'.format(ut.repr3(actor.infr_config)))

        # Configure
        for key in actor.infr_config:
            actor.infr.params[key] = actor.infr_config[key]

        actor.infr.print('infr.status() = {}'.format(ut.repr4(actor.infr.status())))

        # Load models for verifier
        actor.infr.print('loading published models')
        actor.infr.load_published()

    def _find_lnbnn_candidate_edges(
        actor,
        qaids,
        desired_states=[UNREV],
        can_match_samename=False,
        can_match_sameimg=False,
        K=5,
        Knorm=5,
        requery=True,
        prescore_method='csum',
        score_method='csum',
        sv_on=True,
        cfgdict_=None,
        batch_size=None,
    ):
        """
        Copy of similarly-named function from the infr class.
        Copied here so that we can separate query ids from database ids.
        """
        # TODO: abstract into a Ranker class

        # do LNBNN query for new edges
        # Use one-vs-many to establish candidate edges to classify
        cfgdict = {
            'resize_dim': 'width',
            'dim_size': 700,
            'requery': requery,
            'can_match_samename': can_match_samename,
            'can_match_sameimg': can_match_sameimg,
            'K': K,
            'Knorm': Knorm,
            'sv_on': sv_on,
            'prescore_method': prescore_method,
            'score_method': score_method,
        }

        print('[find_lnbnn_candidate_edges] Using cfgdict = {}'.format(ut.repr3(cfgdict)))

        ranks_top = actor.infr.params['ranking.ntop']
        response = actor.infr.exec_matching(
            qaids=qaids,
            name_method='wbia',  # was 'edge',
            cfgdict=cfgdict,
            batch_size=batch_size,
            ranks_top=ranks_top,
        )

        assert response is not None
        lnbnn_results = set(response)

        candidate_edges = {
            edge
            for edge, state in zip(
                lnbnn_results, actor.infr.edge_decision_from(lnbnn_results)
            )
            if state in desired_states
        }

        # actor.infr.print(
        print(
            'ranking alg found {}/{} {} edges'.format(
                len(candidate_edges), len(lnbnn_results), desired_states
            ),
            1,
        )

        return candidate_edges

    def run_ranker_lnbnn(actor, aids, dbdir, qaids=None, config=None, **kwargs):
        """
        Run Hotspotter on the queries, against the aids database.
        Store the resulting candidate matches as UNREVIEWED reviews.
        """
        actor._init_infr(aids, dbdir, **kwargs)

        # If no queries are provided run all against themselves.
        if qaids is None:
            qaids = aids

        # Set the ranker config as an updated copy of the default config
        # The default config is not changed
        if config is not None:
            ranker_config = actor.ranker_config.copy()
            ranker_config.update(config)
        else:
            ranker_config = actor.ranker_config

        # Do the actual work of LNBNN
        candidate_edges = []
        K = 5
        Knorm = 5
        desired_states = [POSTV, NEGTV, INCMP, UNKWN, UNREV]  # prob reduce to just UNREV
        for score_method in ['csum']:  # #['csum', 'nsum']:
            candidate_edges += actor._find_lnbnn_candidate_edges(
                qaids,
                desired_states=desired_states,
                can_match_samename=True,
                K=K,
                Knorm=Knorm,
                prescore_method=score_method,
                score_method=score_method,
                requery=False,
                cfgdict_=ranker_config,
            )
            candidate_edges += actor._find_lnbnn_candidate_edges(
                qaids,
                desired_states=desired_states,
                can_match_samename=False,
                K=K,
                Knorm=Knorm,
                prescore_method=score_method,
                score_method=score_method,
                requery=False,
                cfgdict_=ranker_config,
            )

        # Record the candidate edges that aren't already among the reviews
        ibs = actor.infr.ibs
        prev_rowids = ibs.get_review_rowids_between(aids)
        prev_review_prs = ibs.get_review_aid_tuple(prev_rowids)
        candidate_edges = list(set(candidate_edges) - set(prev_review_prs))

        aids1 = [min(m, n) for m, n in candidate_edges]
        aids2 = [max(m, n) for m, n in candidate_edges]
        num_edges = len(candidate_edges)
        decisions = [UNREVIEWED] * num_edges
        identity_list = [HUMAN_IDENTITY] * num_edges
        ibs.add_review(aids1, aids2, decisions, identity_list=identity_list)

        logger.info(f'LNBNN generated {len(candidate_edges)} candidate edges')

        actor.infr = None  # Delete the infer

    def _prepare_reviews_and_edges(actor):
        """
        Prepare the reviews and the edge weights for LCA. In the usual case, where
        the edge weighter calibration reviews are available, this function will end
        after step 4. The remaining steps are for setting up the process of
        gathering calibration reviews.
        """
        #  0. Make sure this function is called only once for each actor object
        assert actor.new_verifier_quads is None

        logger.info('At the start of _prepare_reviews_and_edges')

        '''
        0. Gather information about the current state of reviews and
           edge weights. Gathering it here, in one place, simplifies
           the rest of the function.
        '''
        ibs = actor.infr.ibs
        aids = actor.infr.aids

        # 0.a reviews
        rev_rowids = ibs.get_review_rowids_between(aids)
        edge_pairs = ibs.get_review_aid_tuple(rev_rowids)
        rev_decisions = ibs.get_review_decision(rev_rowids)
        rev_is_positive = [dec == POSITIVE for dec in rev_decisions]
        rev_is_negative = [dec == NEGATIVE for dec in rev_decisions]
        rev_is_unreviewed = [dec == UNREVIEWED for dec in rev_decisions]
        rev_pos_edges = ut.compress(edge_pairs, rev_is_positive)
        rev_neg_edges = ut.compress(edge_pairs, rev_is_negative)
        rev_unrev_edges = ut.compress(edge_pairs, rev_is_unreviewed)
        logger.info(
            f'Among {len(rev_rowids)} reviews, there are '
            + f'{len(rev_pos_edges)} positive, '
            + f'{len(rev_neg_edges)} negative, and '
            + f'{len(rev_unrev_edges)} unreviewed pairs'
        )

        # 0.b Edge weights
        ew_rowids = ibs.get_edge_weight_rowids_from_edges(edge_pairs)
        ew_rowids = ut.flatten(ew_rowids)
        ew_identities = ibs.get_edge_weight_identity(ew_rowids)
        ew_is_algo = [is_identity_algo(id) for id in ew_identities]
        ew_is_human = [is_identity_human(id) for id in ew_identities]
        ew_algo_rowids = ut.compress(ew_rowids, ew_is_algo)
        ew_human_rowids = ut.compress(ew_rowids, ew_is_human)
        ew_algo_edges = ibs.get_edge_weight_aid_tuple(ew_algo_rowids)
        ew_algo_edges = sorted(list(set(ew_algo_edges)))
        ew_human_edges = ibs.get_edge_weight_aid_tuple(ew_human_rowids)
        ew_human_edges = sorted(list(set(ew_human_edges)))
        logger.info(
            f'Among {len(ew_rowids)} edge weight rowids: '
            + f'{len(ew_algo_edges)} unique verification algorithm edges, and '
            + f'{len(ew_human_edges)} unique human decision edges'
        )

        '''
        1. Form the candidate verifier algorithm edges. Such an edge is ANY review
           edge that is not already in the edge_weights as a verification algorithm edge.
        '''
        cand_algo_edges = list(set(edge_pairs) - set(ew_algo_edges))
        logger.info(f'Step 1: yields {len(cand_algo_edges)} new verifier algorithm edges')

        '''
        2. Compute verifier edge probabilities for the candidate edges and then
           form into a list of edge probability quads (using LCA node ids). Then
           form the edges (as WBIA aids) into prob dictionary
        '''
        logger.info('Step 2: Computing these verifier probabilities. May be a while')
        cand_probs = actor._candidate_edge_probs(cand_algo_edges)
        zipped = list(zip(cand_algo_edges, cand_probs))
        actor.new_verifier_quads = [
            (
                convert_wbia_annot_id_to_lca_node_id(e[0]),
                convert_wbia_annot_id_to_lca_node_id(e[1]),
                p,
                ALGO_AUG_NAME,
            )
            for e, p in zipped
        ]
        actor.probs_for_review_edges = {e: p for e, p in zipped}
        logger.info(f'Step 2: Formed {len(actor.new_verifier_quads)} verifier edge quads')

        '''
        3. Any completed reviews that are NOT in the edge weights become
           new triples that will get sent to direct LCA's focus
        '''
        lca_pos_edges = list(set(rev_pos_edges) - set(ew_human_edges))
        lca_neg_edges = list(set(rev_neg_edges) - set(ew_human_edges))
        actor.new_human_review_triples = [
            (
                convert_wbia_annot_id_to_lca_node_id(a1),
                convert_wbia_annot_id_to_lca_node_id(a2),
                False,
            )
            for a1, a2 in lca_neg_edges
        ]
        actor.new_human_review_triples += [
            (
                convert_wbia_annot_id_to_lca_node_id(a1),
                convert_wbia_annot_id_to_lca_node_id(a2),
                True,
            )
            for a1, a2 in lca_pos_edges
        ]
        logger.info(
            'Step 3 initialized new_human_triples for LCA: '
            + f'{len(lca_neg_edges)} negative reviews and'
            + f'{len(lca_pos_edges)} positive reviews.'
        )

        '''
        4. Check if we have LCA weighter calibration data already. Unless this is
           a new species (newly using LCA), this should usually be the case in
           practice.
        '''
        verifier_calib_filepath = actor.overall_config.get(
            'verifier_calib_filepath', None
        )
        if verifier_calib_filepath is not None:
            if os.path.isfile(verifier_calib_filepath):
                actor.reviews_for_LCA_calib = ut.load_cPkl(verifier_calib_filepath)
                actor.LCA_calib_reviews_are_from_file = True
                num_pos = len(
                    actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_positive_probs']
                )
                num_neg = len(
                    actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_negative_probs']
                )
                logger.info(
                    f'Step 4: loading LCA calibration probs from {verifier_calib_filepath}'
                )
                logger.info(
                    f'Loaded {num_pos} positive review probs from {verifier_calib_filepath}'
                )
                logger.info(
                    f'Loaded {num_neg} negative review probs from {verifier_calib_filepath}'
                )
                return
            else:
                logger.info(
                    f'Step 4: no file {verifier_calib_filepath} for LCA human review probs for wgtr'
                )

        '''
        5. At this point we know we don't have the file of human reviews probabilities
           for LCA weighter calibration. Therefore we need to gather human review
           pairs, starting with the ones that are already completed.
        '''

        # 5a. Compute the probabilities for these review edges and save for calibration.
        #     Note that these may have already been computed above, but since the
        #     calibration set is typically small, I'm not worred about this for now.
        new_pos_probs = actor._candidate_edge_probs(rev_pos_edges)
        new_neg_probs = actor._candidate_edge_probs(rev_neg_edges)
        actor.reviews_for_LCA_calib = {
            ALGO_AUG_NAME: {
                'gt_positive_probs': new_pos_probs,
                'gt_negative_probs': new_neg_probs,
            }
        }

        # 5b. Determine the number of positive and negative edges still needed
        num_req = actor.config["LCA_calib_required_reviews"]
        num_pos_needed = max(0, num_req - len(rev_pos_edges))
        num_neg_needed = max(0, num_req - len(rev_neg_edges))
        logger.info(
            f'Step 5: {num_pos_needed} positive reviews still needed '
            + f'and {num_neg_needed} negative reviews still needed'
        )

        # 5c. If there are enough we stop
        if num_pos_needed == 0 and num_neg_needed == 0:
            return

        '''
        6. Finally, we organize the unreviewed edges in preparation to get
           more human reviews for LCA weighter calibration. We do this by
           pairing unreviewed edges and their probabilities, binning these
           in increments of 0.1 probability, and then pulling one value
           from each bin, in order, until all bins are empty
        '''
        # 6a. Start by gathering all unreviewed edges and their verifier probs.
        edges_needing_prob = [
            e for e in rev_unrev_edges if e not in actor.probs_for_review_edges
        ]
        edge_probs = actor._candidate_edge_probs(edges_needing_prob)
        new_dict = {e: p for e, p in zip(edges_needing_prob, edge_probs)}
        actor.probs_for_review_edges.update(new_dict)
        rev_unrev_probs = [actor.probs_for_review_edges[e] for e in rev_unrev_edges]
        num_bins = 10
        quantized_probs = [int(num_bins * p) for p in rev_unrev_probs]
        logger.info(f'Step 5a: Num unreviewed edges for calib {len(rev_unrev_edges)}')

        #  6b. Apply a quick sanity check on the availability
        if len(rev_unrev_edges) < (num_pos_needed + num_neg_needed):
            logger.info('Error: there are not enough reviews for calibration')
        elif len(rev_unrev_edges) < 2 * (num_pos_needed + num_neg_needed):
            logger.info('Warning: there may not be enough reviews for calibration')

        #  6c. Put the probability / edge pairs into their bins
        bins = [[] for _ in range(num_bins)]
        for p, e in zip(quantized_probs, rev_unrev_edges):
            p = min(p, num_bins - 1)
            bins[p].append(e)

        #  6d. Shuffle each bin
        for i in range(num_bins):
            random.shuffle(bins[i])

        #  6e. In each iteration either delete the current bin because it
        #      is empty or pull from the back of the current bin.  Then, move
        #      on to the next bin.
        before = []
        i = 0
        while len(bins) > 0:
            if len(bins[i]) == 0:
                del bins[i]
                if i == len(bins):
                    i = 0
            else:
                before.append(bins[i][-1])
                del bins[i][-1]
                i = (i + 1) % len(bins)

        #  6f. Final assignment of reviews
        actor.before_review_for_LCA_calib = before
        actor.in_review_for_LCA_calib = []
        logger.info(
            'Step 6, end of _prepare_reviews_and_edges: '
            + f'{len(actor.before_review_for_LCA_calib)} unreviewed edges'
        )

    def _add_feedback_during_LCA_calib(
        actor,
        edge,
        evidence_decision=None,
        tags=None,
        user_id=None,
        meta_decision=None,
        confidence=None,
        timestamp_c1=None,
        timestamp_c2=None,
        timestamp_s1=None,
        timestamp=None,
        verbose=None,
        priority=None,
    ):
        logger.info(
            f'add_feedback_during_LCA_calib: edge {edge}, decision {evidence_decision}'
        )

        """
        Since we are in calibration mode, we only want one completed review
        per annotation pair (edge). Hence, if one already exists, don't add
        a second (there should only be one anyway, but sometimes there are delays)
        to avoid unnecessary duplication.
        """

        ibs = actor.infr.ibs
        rowids = ut.flatten(ibs.get_review_rowids_from_edges([edge]))
        decisions = ibs.get_review_decision(rowids)
        is_known = [d in DECISION_TYPES for d in decisions]
        is_unknown = [not known for known in is_known]

        logger.info(
            f'For this edges, we already have rowids {rowids}, decisions {decisions}'
        )

        #  Remove reviews marked as unknown / unreviewed from the DB
        if any(is_unknown):
            to_remove = ut.compress(rowids, is_unknown)
            logger.info(f'Removing {len(to_remove)} unknown/unreviewed')
            ibs.delete_review(to_remove)

        #  Remove from the list of edges awaiting reviews
        if edge in actor.in_review_for_LCA_calib:
            logger.info(f'Removing from list in review {edge}')
            actor.in_review_for_LCA_calib.remove(edge)

        #  If there is already a known review there will be at least two, so delete all but the first.
        if sum(is_known) > 1:
            logger.info('Repeated calibration review, so deleting and not using again')
            rev_rowids = ut.compress(rowids, is_known)
            ibs.delete_review(rev_rowids[1:])
            return

        #  Get the WBIA aids and LCA node name
        a1, a2 = edge
        node1 = convert_wbia_annot_id_to_lca_node_id(a1)
        node2 = convert_wbia_annot_id_to_lca_node_id(a2)

        #  Record the probability associated with the decision for LCA weight
        #  calibration and form the triple that will be sent to LCA
        prob = actor.probs_for_review_edges.get(edge, -1)
        logger.info(f'prob {prob}')

        if prob == -1:
            logger.info('Not a candidate edge so doing nothing')
            return
        if evidence_decision == POSTV:
            actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_positive_probs'].append(prob)
            triple = (node1, node2, True)
        elif evidence_decision == NEGTV:
            actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_negative_probs'].append(prob)
            triple = (node1, node2, False)
        else:
            logger.info('decision is neither pos nor neg so doing nothing')
            return

        #  Save the triple for LCA.
        logger.info(f'Adding triple {triple} for LCA and recording review in DB')
        actor.new_human_review_triples.append(triple)

    def _init_weighter(actor):
        """
        Check to see if there are enough reviews and if so, initialize the weighter
        """
        n_required = actor.config['LCA_calib_required_reviews']
        pos_probs = actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_positive_probs']
        neg_probs = actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_negative_probs']

        logger.info(
            f'_init_weighter: {n_required} are required, currently have '
            f'{len(pos_probs)} positive and {len(neg_probs)} negative'
        )
        if not actor.LCA_calib_reviews_are_from_file and (
            len(pos_probs) < n_required or len(neg_probs) < n_required
        ):
            logger.info('Not enough so returning without initializing weighter')
            return False

        """
        We have enough reviews!  If these are not already from a file, then save them
        to the given file.
        """
        if not actor.LCA_calib_reviews_are_from_file:
            logger.info('Not from file')
            pos_probs = filter_prob_outliers(pos_probs)
            logger.info(f'After filtering there are {len(pos_probs)} positive remaining')
            actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_positive_probs'] = pos_probs
            neg_probs = filter_prob_outliers(neg_probs)
            logger.info(f'After filtering there are {len(neg_probs)} positive remaining')
            actor.reviews_for_LCA_calib[ALGO_AUG_NAME]['gt_negative_probs'] = neg_probs

            verifier_calib_filepath = actor.overall_config.get(
                'verifier_calib_filepath', None
            )
            if verifier_calib_filepath is not None:
                ut.save_cPkl(verifier_calib_filepath, actor.reviews_for_LCA_calib)
                logger.info(
                    f'Saved verifier human review GT probs to {verifier_calib_filepath}'
                )
            else:
                logger.info(
                    'Did not save verifier human review GT probs because filepath is None'
                )

        logger.info('Here are the LCA probs for weight calibration')
        logger.info(ut.repr3(actor.reviews_for_LCA_calib[ALGO_AUG_NAME]))

        """
        Actually generate the weighters.
        """
        wgtrs = ga_driver.generate_weighters(
            actor.lca_config, actor.reviews_for_LCA_calib
        )
        actor.wgtr = wgtrs[0]

        # Update delta score thresholds in the lca configg
        multiplier = actor.lca_config['min_delta_converge_multiplier']
        ratio = actor.lca_config['min_delta_stability_ratio']

        human_gt_positive_weight = actor.wgtr.human_wgt(is_marked_correct=True)
        human_gt_negative_weight = actor.wgtr.human_wgt(is_marked_correct=False)

        human_gt_delta_weight = human_gt_positive_weight - human_gt_negative_weight
        convergence = -1.0 * multiplier * human_gt_delta_weight
        stability = convergence / ratio

        actor.lca_config['min_delta_score_converge'] = convergence
        actor.lca_config['min_delta_score_stability'] = stability

        logger.info(
            'Using provided   min_delta_converge_multiplier = %0.04f' % (multiplier,)
        )
        logger.info('Using provided   min_delta_stability_ratio     = %0.04f' % (ratio,))
        logger.info(
            'Using calculated min_delta_score_converge      = %0.04f' % (convergence,)
        )
        logger.info(
            'Using calculated min_delta_score_stability     = %0.04f' % (stability,)
        )
        return True

    def _init_lca(actor):
        """
        Initialize the weighter and if it succeeds, also initialize the db and
        the edge generator.
        """
        wgtr_initialized = actor._init_weighter()
        if not wgtr_initialized:
            return

        actor.db = db_interface_wbia(actor)
        actor.edge_gen = edge_generator_wbia(actor.db, actor.wgtr, controller=actor)
        actor.has_LCA_calib = True

    """  *****************************************
    The next three functions compute the probabilities for edge weighting
    """

    def _candidate_edge_probs_auto(actor, candidate_edges):
        # Use the VAMP verifier
        task_probs = actor.infr._make_task_probs(candidate_edges)
        match_probs = list(task_probs['match_state']['match'])
        nomatch_probs = list(task_probs['match_state']['nomatch'])
        candidate_probs = []
        for match_prob, nomatch_prob in zip(match_probs, nomatch_probs):
            prob_ = 0.5 + (match_prob - nomatch_prob) / 2
            candidate_probs.append(prob_)
        return candidate_probs

    def _candidate_edge_probs_pie_v2(actor, candidate_edges):
        # Use the PIE verifier
        from wbia_pie_v2._plugin import distance_to_score

        # Ensure features
        actor.infr.ibs.pie_v2_embedding(actor.infr.aids)

        candidate_probs = []
        for edge in tqdm.tqdm(candidate_edges):
            qaid, daid = edge
            pie_annot_distances = actor.infr.ibs.pie_v2_predict_light_distance(
                qaid,
                [daid],
            )
            pie_annot_distance = pie_annot_distances[0]
            score = distance_to_score(pie_annot_distance, norm=500.0)
            candidate_probs.append(score)

        return candidate_probs

    def _candidate_edge_probs(actor, candidate_edges):
        """
        Given the candidate edges this returns:
        . Their verifier probabilities
        This does NOT change the LCA database.
        """
        if len(candidate_edges) == 0:
            return []

        verifier_algo = actor.verifier_config.get('verifier', 'vamp')
        if verifier_algo == 'vamp':
            candidate_probs = actor._candidate_edge_probs_auto(candidate_edges)
        elif verifier_algo == 'pie_v2':
            candidate_probs = actor._candidate_edge_probs_pie_v2(candidate_edges)
        else:
            raise ValueError('Verifier algorithm %r is not supported' % (verifier_algo,))

        num_probs = len(candidate_probs)
        if num_probs == 0:
            args = [num_probs] + ['None'] * 3
        else:
            min_probs = '%0.04f' % (min(candidate_probs),)
            max_probs = '%0.04f' % (max(candidate_probs),)
            avg_probs = '%0.04f' % (sum(candidate_probs) / num_probs)
            args = (num_probs, min_probs, max_probs, avg_probs)
        logger.info('Verifier probs on %d edges: min %s, max %s, avg %s' % args)
        return candidate_probs

    """  *****************************************
    The next set of functions is for simulation of the review procedure
    and for short-circuiting of ground truth.
    """

    def _load_simulation_ground_truth(actor):
        if (
            not actor.config.get('simreview.enabled')
            or actor.failed_to_open_gt_clusters_filepath
        ):
            return

        clustering_gt_filepath = actor.config.get('clustering_gt_filepath', None)
        if clustering_gt_filepath is None or not os.path.isfile(clustering_gt_filepath):
            actor.failed_to_open_gt_clusters_filepath = True
            logger.info(
                'Requesting simulation but could not open gt aids '
                + f'cluster file {clustering_gt_filepath}'
            )
        else:
            actor.gt_aid_clusters = ut.load_cPkl(clustering_gt_filepath)
            gt_aids = list(actor.gt_aid_clusters.keys())
            num_aids = len(actor.gt_aid_clusters)
            num_nids = len(set(actor.gt_aid_clusters.values()))
            logger.info(
                f'Loaded {num_aids} aids and {num_nids} nids/clusters '
                + f'from {clustering_gt_filepath}'
            )
            if not (set(gt_aids) == set(actor.infr.aids)):
                not_in_infr_aids = set(gt_aids) - set(actor.infr.aids)
                not_in_gt_aids = set(actor.infr.aids) - set(gt_aids)
                logging.info("ERROR: in simulation, aids different from GT aids")
                logging.info(f'{len(not_in_infr_aids)} missing from aids')
                logging.info(f'{len(not_in_gt_aids)} missing from GT aids')
                assert False

    def _attempt_short_circuit(actor, edges):
        if not actor.overall_config.get('use_short_circuiting', False):
            return edges
        ibs = actor.infr.ibs

        remaining_edges = []
        for edge in edges:
            if should_short_circuit_review(ibs, edge):
                aid1, aid2 = edge
                if actor.gt_aid_clusters is None:
                    actor.short_circuit_count += 1
                    logger.info(
                        f'short-circuiting {edge}, count is {actor.short_circuit_count}'
                    )
                elif actor.gt_aid_clusters[aid1] == actor.gt_aid_clusters[aid2]:
                    actor.short_circuit_incorrect += 1
                    logger.info(
                        f'incorrectly short-circuiting {edge}; '
                        + f'num mistakes {actor.short_circuit_incorrect}'
                    )
                else:
                    actor.short_circuit_correct += 1
                    logger.info(
                        f'correctly short-circuiting {edge}; '
                        + f'num correct {actor.short_circuit_correct}'
                    )
                feedback = {
                    'edge': edge,
                    'user_id': SHORTCIRCUIT_REVIEW_IDENTITY,
                    'confidence': const.CONFIDENCE.CODE.PRETTY_SURE2,
                    'evidence_decision': NEGTV,
                }
                actor.feedback(**feedback)
            else:
                actor.no_short_circuit_count += 1
                logger.info(
                    f'No short-circuiting {edge} '
                    + f'total {actor.no_short_circuit_count}'
                )
                remaining_edges.append(edge)

        return remaining_edges

    def _attempt_simulation_reviews(actor, edges):
        if (
            not actor.config.get('simreview.enabled')
            or actor.failed_to_open_gt_clusters_filepath
        ):
            return edges
        assert actor.gt_aid_clusters is not None

        logger.info(f'Attempting simulation reviews for {len(edges)} edges')
        for edge in edges:
            aid1, aid2 = edge
            clusters = actor.gt_aid_clusters
            if not (aid1 in clusters and aid2 in clusters):
                logger.info(f'Warning: edge ({aid1},{aid2}) not in simulation GT')
                continue
            is_same_name = clusters[aid1] == clusters[aid2]

            prob_human_correct = actor.config.get('simreview.prob_human_correct')
            rand = random.uniform(0.0, 1.0)
            make_mistake = rand > prob_human_correct

            if is_same_name and not make_mistake:
                decision = POSTV
                logger.info(f'edge {edge}, same GT cluster and no mistake so POSTV')
            elif is_same_name and make_mistake:
                decision = NEGTV
                logger.info(f'edge {edge}, same GT cluster but making mistake so NEGTV')
            elif make_mistake:
                decision = POSTV
                logger.info(
                    f'edge {edge}, not same GT cluster but making mistake so POSTV'
                )
            else:
                decision = NEGTV
                logger.info(f'edge {edge}, not same GT cluster and no mistake so NEGTV')

            feedback = {
                'edge': edge,
                'user_id': SIMREVIEW_IDENTITY,
                'confidence': const.CONFIDENCE.CODE.PRETTY_SURE,
                'evidence_decision': decision,
            }
            actor.feedback(**feedback)

        return None

    def _short_circuit_and_simulation(actor, edges):
        """
        Attempt to short circuit human reviews. Failing this, try to use a
        simulation of the ground truth reviews.  Failling this, send the edges
        back for "normal" processing. This function starts by loading
        """
        actor._load_simulation_ground_truth()
        edges = actor._attempt_short_circuit(edges)
        if edges is not None:
            edges = actor._attempt_simulation_reviews(edges)
        return edges

    def _make_review_tuples(actor, edges):
        """
        Add information about the edges to form the review requests.
        This does nothing other than to use the defaults.
        """
        if edges is None:
            return None

        priority = 1.0
        review_tuples = []
        for edge in edges:
            edge_data = actor.infr.get_nonvisual_edge_data(edge, on_missing='default')
            # Extra information
            edge_data['nid_edge'] = None
            if actor.edge_gen is None:
                edge_data['queue_len'] = 0
            else:
                edge_data['queue_len'] = len(actor.edge_gen.edge_requests)
            edge_data['n_ccs'] = (-1, -1)
            review_tuples.append((edge, priority, edge_data))
        return review_tuples

    def start(actor, dbdir, aids, config={}, graph_uuid=None, **kwargs):
        logger.info(f'In start with {len(aids)} aids')
        actor.config.update(config)

        # Initialize inference object
        actor._init_infr(aids, dbdir, **kwargs)
        actor.graph_uuid = graph_uuid

        # Initialize the review iterator
        actor._gen = actor.main_gen()

        # CHANGE ME AFTER OTHER DEBUGGING
        status = 'warmup'

        logger.info(f'Leaving start with status {status}')
        return status

    def save_clustering_separately(actor, fp):
        """
        Save the clustering as a pickled dictionary mapping from annotation id to name id.
        """
        if fp is None:
            return
        ibs = actor.infr.ibs
        aids = actor.infr.aids
        nids = ibs.get_annot_nids(aids)
        num_nids = len(set(nids))
        aids_to_nids = {a: n for a, n in zip(aids, nids)}
        ut.save_cPkl(fp, aids_to_nids)
        logger.info(
            f'Saved mapping of {len(aids)} annot ids (aids) to {num_nids} '
            + f' name ids / nids to {fp} '
        )

    def main_gen(actor):
        assert actor.infr is not None

        actor.phase = 0
        actor.loop_phase = 'warmup'

        logger.info('Entering actor.main_gen for the first time!')

        '''
        1. Organize the candidate edges, and the human reviews for LCA.
           Then try to initialize the weighter and database and edge
           generator. The key is having enough reviews for the weighter.
        '''
        actor._prepare_reviews_and_edges()
        actor._init_lca()

        '''
        2. If we have already gotten enough reviews for LCA weighter calibration
           then we need to get them from the human reviewer(s). The main effort
           of this loop is to keep feeding requests. Note that the feedback from
           the requests is handled by the actor.add_feedback_during_LCA_calib
           function above.
        '''
        while not actor.has_LCA_calib:
            logger.info('main_gen: getting reviews for calibration')
            assert actor.reviews_for_LCA_calib is not None

            target_num = actor.config['LCA_calib_num_to_queue']
            num_to_add = max(0, target_num - len(actor.in_review_for_LCA_calib))
            num_to_add = min(num_to_add, len(actor.before_review_for_LCA_calib))
            logger.info(
                f'{target_num} target, {len(actor.in_review_for_LCA_calib)} in review'
            )
            logger.info(
                f'{len(actor.before_review_for_LCA_calib)} before review, {num_to_add} to add'
            )
            edges_to_add = actor.before_review_for_LCA_calib[:num_to_add]
            actor.before_review_for_LCA_calib = actor.before_review_for_LCA_calib[
                num_to_add:
            ]
            edges_to_add = actor._short_circuit_and_simulation(edges_to_add)

            if edges_to_add is not None:
                logger.info(
                    f'Num to add after attempting short circuit / sim review {len(edges_to_add)}'
                )
                actor.in_review_for_LCA_calib += edges_to_add
                if len(edges_to_add) == 1:
                    e = edges_to_add[0]
                    prob = actor.probs_for_review_edges.get(e, None)
                    logger.info(f'Edge prob: edge {e}, prob {prob}')
                else:
                    edge_probs = {
                        e: actor.probs_for_review_edges.get(e, None) for e in edges_to_add
                    }
                    logger.info(f'Edge probs: {ut.repr3(edge_probs)}')
                user_requests = actor._make_review_tuples(actor.in_review_for_LCA_calib)
                logger.info(
                    f'Yielding human review requests of length {len(user_requests)}'
                )
                yield user_requests

            # Attempt initialization again.
            actor._init_lca()

        """
        3. Now we have LCA weighter calibration complete and other classes initialized,
           we are ready to star
        """
        actor.phase = 1
        actor.loop_phase = 'driver'

        #  3a. Set the verifier pairs (from the ranker)
        verifier_results = actor.new_verifier_quads
        if verifier_results is None:
            verifier_results = []
        logger.info(f'Sending {len(verifier_results)} new verifier quads to ga_driver')

        #  3b. Set the human-reviewed edges for adding to the graph
        human_triples = actor.new_human_review_triples
        if human_triples is None:
            human_triples = []
        logger.info(f'Sending {len(human_triples)} new human review triples to ga_driver')

        #  3c. In the atypical case that both of the foregoing are empty, it is likely
        #      that LCA processing was stopped before completion, but after adding
        #      the verifier and/or human results.  To recover from this, we tell
        #      LCA to review the clusters and work from there.  The run_on_all_names is
        #      get to True to force everything to be checked.
        if (
            len(verifier_results) == 0 and len(human_triples) == 0
        ) or actor.overall_config['run_on_all_names']:
            cluster_ids_to_check = list(actor.db.clustering.keys())
            logger.info(
                'Starting with no new verifier or human review edges, '
                + f'so re-examining all {len(cluster_ids_to_check)} clusters'
            )
        else:
            cluster_ids_to_check = []

        #  3d. Form the driver.  This computes the ccPICs.
        actor.driver = ga_driver.ga_driver(
            verifier_results,
            human_triples,
            cluster_ids_to_check,
            actor.db,
            actor.edge_gen,
            actor.lca_config,
        )

        """
        4. Set up the run_all_ccPICs iterator
        """
        actor.phase = 2
        actor.loop_phase = 'run_all_ccPICs'
        partial_progress_cb = partial(progress_db, actor)
        actor.ga_gen = actor.driver.run_all_ccPICs(
            yield_on_paused=True,
            progress_cb=partial_progress_cb,
            # other_clustering=other_clustering,
        )

        """
        4. Run the main loop of generating the clustering one ccPIC at a time.
        """
        changes_to_review = []
        while True:
            try:
                # 4a. Run the next step. If the yield is a StopIteration exception
                #     then the LCA clustering work is done and we proceed to committing
                #     The changes below
                change_to_review = next(actor.ga_gen)
            except StopIteration:
                break

            if change_to_review is None:
                # 4b. If the change_to_review is None we are in the middle of LCA applied
                #     to a single CCPIC and more human reviews are needed.
                requested_edges = []
                for edge in actor.edge_gen.get_edge_requests():
                    n0, n1, aug_name = edge
                    assert is_aug_name_human(aug_name)
                    aid1 = convert_lca_node_id_to_wbia_annot_id(n0)
                    aid2 = convert_lca_node_id_to_wbia_annot_id(n1)
                    requested_edges.append((aid1, aid2))

                logger.info(f'Received {len(requested_edges)} human review requests')

                requested_edges = actor._short_circuit_and_simulation(requested_edges)
                user_requests = actor._make_review_tuples(requested_edges)

                #  Yield all waiting edges awaiting human review
                if user_requests is not None:
                    logger.info(f'Yielding {len(user_requests)} user_requests')
                    yield user_requests
            else:
                #  4c. The third case for the end of an iteration (a yield from
                #      run_all_ccPICs) is the completion of a single ccPIC. In this
                #      case the cluster changes from the ccPIC are return for review
                #      and commitment.
                changes_to_review.append(change_to_review)

        """
        5.  Finally, review and commit the changes to the WBIA database. This will
            require significant modification when human approval of final changes
            is instituted.
        """
        actor.phase = 3
        actor.loop_phase = 'commit_cluster_change'
        actor.changes = []
        c_count = 0
        for changes in changes_to_review:
            for cc in changes:
                change = actor.db.commit_cluster_change(cc)
                logger.info(f'Cluster change {c_count} for WBIA')
                if change is None:
                    logger.info('None!')
                elif isinstance(change, dict):
                    for k, v in change.items():
                        logger.info(f'   {k}: {v}')
                else:
                    logger.info(f'type is {type(change)}')
                c_count += 1
                if change is not None:
                    actor.changes.append(change)

        '''
        6.  After committing the changes, save the current association between
            aids and nids outside the wbia database.
        '''
        fp = actor.config.get('clustering_result_filepath', None)
        actor.save_clustering_separately(fp)

        #  7.  Done!
        actor.phase = 4
        actor.loop_phase = None
        return 'finished:main'

    def resume(actor):
        with actor.resume_lock:
            if actor._gen is None:
                return 'finished:stopped'
            try:
                user_request = next(actor._gen)
            except StopIteration:
                actor._gen = None
                user_request = 'finished:stopiteration'
            return user_request

    def feedback(actor, **feedback):
        actor.infr.add_feedback(**feedback)
        actor.infr.write_wbia_staging_feedback()
        if actor.has_LCA_calib:
            actor.edge_gen.add_feedback(**feedback)
        else:
            actor._add_feedback_during_LCA_calib(**feedback)

    def add_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def remove_aids(actor, aids, **kwargs):
        raise NotImplementedError()

    def logs(actor):
        return None

    def status(actor):
        actor_status = {}
        try:
            actor_status['phase'] = actor.phase
        except Exception:
            pass
        try:
            actor_status['loop_phase'] = actor.loop_phase
        except Exception:
            pass
        try:
            actor_status['is_inconsistent'] = False
        except Exception:
            pass
        try:
            actor_status['is_converged'] = actor.phase > 4
        except Exception:
            pass
        try:
            actor_status['num_meaningful'] = 0
        except Exception:
            pass
        try:
            actor_status['num_pccs'] = (
                None if actor.edge_gen is None else len(actor.edge_gen.edge_requests)
            )
        except Exception:
            pass
        try:
            actor_status['num_inconsistent_ccs'] = 0
        except Exception:
            pass
        try:
            actor_status['cc_status'] = {
                'num_names_max': len(actor.db.clustering),
                'num_inconsistent': 0,
            }
        except Exception:
            pass
        try:
            actor_status['changes'] = actor.changes
        except Exception:
            pass
        return actor_status

    def metadata(actor):
        if actor.infr.verifiers is None:
            actor.infr.verifiers = {}
        verifier = actor.infr.verifiers.get('match_state', None)
        extr = None if verifier is None else verifier.extr
        metadata = {
            'extr': extr,
        }
        return metadata


class LCAClient(GraphClient):
    actor_cls = LCAActor

    def sync(client, ibs):
        ret_dict = {
            'changes': client.actor_status.get('changes', []),
        }
        return ret_dict


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_lca._plugin
    """
    # import xdoctest

    # xdoctest.doctest_module(__file__)

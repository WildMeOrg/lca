# -*- coding: utf-8 -*-
import datetime as dt
import networkx as nx
import uuid

import cid_to_lca
import cluster_tools as ct
import draw_lca
import lca
import lca_alg1 as alg1
import lca_alg2 as alg2
import lca_queues
import logging
import weight_manager as wm


logger = logging.getLogger('lca')


"""
Construction:

1. List of weighted edges, each of which is a 4-tuple with
   . node_id0,
   . node_id1,
   . weight,
   . augmentation method name
where the latter is a string giving the name of the augmentation
method used to generate the weight. An augmentation method is either
an automatic verification algorithm or a human reviewer (name
"human").

2. The initial clusters. This is a list of lists of nodes, defaulting
to None. Each list is a cluster of nodes and therefore the
lists must be disjoint. Any node from the list of weighted edges that
is not in a cluster will be placed in its own cluster to start the
computation. Any node from the clusters that is not in the graph will
be added. Thus, the nodes in the graph and the nodes in the clusters
will coincide at all times.

3. List of names of augmentation methods. This list gives the order in
which they are tried.  The list must end with "human", and the length
of the list might only be 1, meaning that no verification algorithms
have been provided and only human input will be used.

4. params - dictionary of tuning parameters and debugging controls.

5. Augmentation request callback: pointer to function that accepts
request for augmentation in the form of a list of 3-tuples. Each
3-tuple contains the two node ids and the name of the augmentation
method.

6. Augmentation result callback: pointer to function that accepts
results of augmentation in the form of a list of 4-tuples. Each
4-tuple contains the two node ids, the wgt and the name of the
augmentation method that produced the result
"""


"""
Non-required Callbacks:

Originally there were a lot of these, but now there are only a few:

1. LCA asks for and receives back nodes to remove from the
graph. These nodes and their associated edges are removed from the
graph.

Important note: what happens if a node is removed from the graph and
then an edge for the node is returned? This can occur due to race
conditions where keep human reviewer deletes an annotation and a
second human reviewer likes the annotation as part of the graph. This
will occur in the next callback. I've currently decided that the node
should stay deleted and therefore that the new edge should not be added

2. Stop check: ask if stop requested needed and stop, returning from
the function called "run_main_loop". A subsequent call to
run_main_loop will pick up cleanly from stop point.

Once the algorithm is stopped, intermediate results and statuses can
be checked and logged. These are implemented as direct method calls,
as outlined below.
"""


"""
Process start: run_main_loop

Runs until either the algorithm has converged, a maximum number of
iterations has been reached, a stop has been requested, or the
algorithm is waiting for too many augmentation edges.

Returns: a 3-tuple containing the following
. pause_for_edges: boolean that will be true if there are too many LCAs
                   waiting for augmentation edges
. iter_num: the ending iteration number.  This allows the iteration
            numbering to continue from where it left off in the
            next call to run_main_loop
. converged: boolean that will be true if the algorithm converged

"""


"""
Notes:

1. The algorithm knows nothing about names (animal ids). It only
reports the clusters that satisfy its optimization criteria up to the
point where it is stopped. It is the responsibility of functionality
outside the algorithm to interpret these results and their impact on
names and therefore ecological entities.

2. We have discussed concerns about users having to do review work for
other contributors if we merge into a large data set.  This is not
necessarily the case: when a user's annotations are entered into the
system and they are run against an indexing (ranking) algorithm, a
subgraph should be constructed from the new annotations and the
ranking matches. Work is focused only on these subgraphs. Moreover,
requests for reviews can be ignored (algorithmically, before they
reach the human) if they don't touch on the contributed annotations.

3. We've discussed cases where a reviewer would like to generate
population estimates from a subset of annotations, as though the
remainder of the annotations were not there. The particular example of
concern is a case where annotation A and B were not found to match,
until annotation C, outside the subset, is introduced. This situation
can be addressed by a partial or complete re-run of the indexing
and verification algorithms focused only on the annotations in the
target subset (including A and B, but not C). When the graph algorithm
is launched and asks for review decisions, cached (manual) decisions
may be used without going back to the reviewer, accelerating the
process by making it less manual. This will produce the estimate as
though C did not exist because it is not included it the graph to form
the hypothesized connection between A and B even though the review
decision for A and B, triggered through C, will be in the
database. Various levels of caching can accelerate this both the
indexing part and the review part.
"""


"""
Data structures:

1. G, the graph itself: a weighted, undirected graph

2. weight_mgr: the object that manages the interface to the
augmentation callbacks, including choosing the appropriate callback.
The rest of the code knows nothing about the choice of augmentation
methods.

3. clustering: cid -> set of nodes; each node is in exactly one set

4. node2cid: graph node -> id of cluster that contains it

5. queues: store all current LCAs on one of Q, S and W

6. cid2lca: manage information about association between one or
cids and the LCAs that might include them.  This is much more
than a mapping from a pair of cids to the LCA that combines them.
"""

"""
Invariants:
1. Each node is in exactly one cluster

2. All graph nodes are in a cluster and all nodes in a cluster are in
   the graph.

3. The algorithm works in three phases: scoring, splitting and
   stability.  During scoring all pairs of clusters joined by (at
   least) one edge form an LCA.  During splitting only individual
   clusters (with at least two nodes) form and LCA.  During stability,
   each individual cluster and each pair of connected clusters form an
   LCA.

4. Each LCA sits in exactly one queue:  the main queue, the scoring
   queue, the waiting queue, and the done queue. The done queue is
   generally only used for LCAs where all "inconsistent" edges have
   been tested too many times and therefore any more tests should be
   considered "futile".
"""


class graph_algorithm(object):  # NOQA
    def __init__(self, edges, clusters, aug_names, params, aug_request_cb, aug_result_cb):
        self.params = params
        logger.info('======================================')
        logger.info('Construction of graph_algorithm object')
        logger.info(dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.weight_mgr = wm.weight_manager(
            aug_names, params['tries_before_edge_done'], aug_request_cb, aug_result_cb
        )
        self.G = nx.Graph()
        weighted_edges = self.weight_mgr.get_initial_edges(edges)
        self.G.add_weighted_edges_from(weighted_edges)
        logger.info(
            'Initial graph has %d nodes and %d edges'
            % (len(self.G.nodes), len(self.G.edges))
        )
        self._next_cid = 0
        self.build_clustering(clusters)
        self.node2cid = ct.build_node_to_cluster_mapping(self.clustering)
        self.score = ct.clustering_score(self.G, self.node2cid)

        self.phase = 'scoring'
        self.cid2lca = cid_to_lca.CID2LCA()
        self.queues = lca_queues.lca_queues()
        self.new_lcas(self.clustering.keys(), use_pairs=True, use_singles=False)
        if self.queues.num_lcas() == 0:
            logger.info("Phase shift immediately into 'splitting'")
            self.phase = 'splitting'
            self.new_lcas(self.clustering.keys(), use_pairs=False, use_singles=True)
        self.queues.info_long(max_entries=10)

        self.num_verifier_results = 0
        self.num_human_results = 0
        self.removed_nodes = set()

        self.should_run_densify = (
            'should_densify' in self.params and self.params['should_densify']
        )

        self.draw_obj = None
        self.ccpic_id = str(uuid.uuid4())
        if self.params['draw_iterations']:
            self.draw_obj = draw_lca.draw_lca(self.params['drawing_prefix'])

        """  Need to set these callbacks to request and receive
        information from the verification algorithm and to do the same
        from human reviewers. """
        self.remove_nodes_cb = None
        self.status_request_cb = None
        self.status_return_cb = None
        self.results_request_cb = None
        self.results_return_cb = None
        self.log_request_cb = None
        self.log_return_cb = None
        self.trace_start_human_gt_cb = None
        self.trace_iter_compare_to_gt_cb = None
        self.should_stop_cb = None
        self.progress_cb = None

        logger.info('Completed graph algorithm initialization')

    def set_remove_nodes_cb(self, cb):
        """
        Callback to indicate nodes to remove from the graph
        """
        self.remove_nodes_cb = cb

    def set_result_cbs(self, request_cb, return_cb):
        """
        Callbacks to return the clustering results.
        """
        self.results_request_cb = request_cb
        self.results_return_cb = return_cb

    def set_log_contents_cbs(self, request_cb, return_cb):
        self.log_request_cb = request_cb
        self.log_return_cb = return_cb

    def set_trace_compare_to_gt_cb(self, start_human, iter_compare):
        self.trace_start_human_gt_cb = start_human
        self.trace_iter_compare_to_gt_cb = iter_compare

    def set_stop_check_cb(self, cb):
        self.stop_check_cb = cb

    def set_progress_cb(self, cb):
        self.progress_cb = cb

    def generate_new_cids(self, k):
        lower = self._next_cid
        self._next_cid = upper = lower + k
        padding = 3
        prefix = 'tc'
        cids = [prefix + str(i).zfill(padding) for i in range(lower, upper)]
        return cids

    def build_clustering(self, clusters):
        """
        From an iterator through the clusters - each of which could be
        a set or a list,
        """
        if clusters is None or len(clusters) == 0:
            # Form an initial clustering where every node starts in its own
            # singleton cluster.
            new_cids = self.generate_new_cids(len(self.G))
            self.clustering = {cid: {n} for cid, n in zip(new_cids, self.G.nodes)}
            logger.info(
                'Initial clusters are singletons with length %d'
                ', formed from the graph nodes' % len(self.clustering)
            )
        else:
            # Form the initial clustering from a previous clustering,
            # and then add clusters from graph nodes that may not be
            # in any clusters.
            new_cids = self.generate_new_cids(len(clusters))
            self.clustering = ct.build_clustering_from_clusters(new_cids, clusters)
            num_previous = len(clusters)

            graph_nodes = set(self.G.nodes())
            cluster_nodes = set.union(*self.clustering.values())
            unclustered = graph_nodes - cluster_nodes

            new_cids = self.generate_new_cids(len(unclustered))
            new_clusters = {i: set([c]) for i, c in zip(new_cids, unclustered)}
            self.clustering.update(new_clusters)

            missing_nodes = cluster_nodes - graph_nodes
            self.G.add_nodes_from(missing_nodes)
            logger.info(
                'Initial clusters include %d from previous clusters, '
                'plus %d new clusters from singletons' % (num_previous, len(new_clusters))
            )
            logger.info('Added %d missing nodes to the graph' % len(missing_nodes))

    def which_lca_types(self):
        from_pairs = self.phase in ['scoring', 'stability']
        from_singles = self.phase in ['splitting', 'stability']
        return from_pairs, from_singles

    def new_lcas(self, new_cids, use_pairs=False, use_singles=False):
        tuples = self.lca_tuples(new_cids, use_pairs, use_singles)
        self.create_lcas_from_tuples(tuples)

    def lca_tuples(self, new_cids, from_pairs, from_singles):
        #  Get the tuples of cids to form LCA, these will be pairs or
        #  individuals or both, depending on the phase of the computation.
        tuples = []
        if from_pairs:
            tuples = ct.form_connected_cluster_pairs(
                self.G, self.clustering, self.node2cid, new_cids
            )
        if from_singles:
            tuples.extend([(cid,) for cid in new_cids if len(self.clustering[cid]) > 1])
        return tuples

    def create_lcas_from_tuples(self, tuples):
        """
        Create LCAs for each tuple, computing the from_score for each,
        forming the actual LCA, adding it to the cid2lca mapping, and
        adding it to the scoring queue.
        """
        num_created = 0
        for cids in tuples:
            if len(cids) == 1:
                nodes = self.clustering[cids[0]]
            else:
                nodes = self.clustering[cids[0]] | self.clustering[cids[1]]
            subG = self.G.subgraph(nodes)
            from_score = ct.cid_list_score(subG, self.clustering, self.node2cid, cids)
            a = lca.LCA(subG, self.clustering, cids, from_score)
            self.cid2lca.add(a)
            self.queues.add_to_S(a)
            num_created += 1
        logger.info('Created %d new LCAs' % num_created)

    def run_main_loop(self, iter_num=0, max_iterations=None):
        halt_requested = False
        should_pause = False
        converged = False

        while (max_iterations is None or iter_num < max_iterations) and not (
            halt_requested or should_pause or converged
        ):
            # if (iter_num >= 540 and iter_num <= 550) or (iter_num > 550 and iter_num % 100 == 0):
            #     graph_algorithm.save_iteration_graph(iter_num)
            iter_num += 1
            logger.info('')
            logger.info('*** ccPIC id is %s, Iteration %d***' % (self.ccpic_id, iter_num))

            
            
            if self.progress_cb is not None:
                self.progress_cb(self, iter_num)

            #  Prepare for the start of the next iteration
            self.remove_nodes()
            self.add_edges()
            self.compute_lca_scores()

            self.show_brief_state()
            if logger.getEffectiveLevel() <= logging.DEBUG:
                self.show_queues_debug()

            """
            Step 2: Handle the top LCA on the main Q, include the
            special case of an empty main queue (top LCA is None)
            which should produce a phase change or a wait for new
            edges, or if the waiting list is empty, convergence.
            """
            a = self.queues.top_Q()
            num_human = self.weight_mgr.num_human_decisions()

            # Step 2a: Apply the LCA if it improves the score:
            if a is not None and a.delta_score() > 0:
                logger.info('Decision: apply LCA')
                # Note: no need to explicitly remove a from the top of
                # the heap because this is done during the replacement
                # process itself.
                self.score += a.delta_score()
                self.apply_lca(a)

            # Step 2b: since the delta is <= 0, if we are in "scoring"
            # switch to splitting
            elif self.phase == 'scoring':
                if self.trace_start_human_gt_cb is not None:
                    self.trace_start_human_gt_cb(self.clustering, self.node2cid, self.G)
                logger.info('Decision: switch phases to splitting')
                self.phase = 'splitting'
                self.queues.switch_to_splitting()
                self.cid2lca.clear()
                self.new_lcas(self.clustering.keys(), use_pairs=False, use_singles=True)
                self.queues.info_long(max_entries=10)
                if self.trace_start_human_gt_cb is not None:
                    self.trace_start_human_gt_cb(self.clustering, self.node2cid, self.G)
                

            # Step 2c: consider shift from splitting to stability
            elif self.phase == 'splitting' and (
                a is None or a.delta_score() < self.params['min_delta_score_stability']
            ):
                logger.info('Decision: switch phases to stability')
                self.phase = 'stability'
                self.queues.switch_to_stability()
                self.new_lcas(
                    self.clustering.keys(), use_pairs=True, use_singles=True
                )  # singles will be kept
                self.queues.info_long(max_entries=10)

            # Step 2d: at this point we should run augmentation if
            # there is still a significant chance of a change
            elif (
                a is not None
                and self.params['min_delta_score_converge'] < a.delta_score()
                and self.should_run_densify
                and num_human >=  self.params["densify_min_number_human"]
            ):
                logger.info(f'ccPIC id is {self.ccpic_id}, Decision: run desnify top LCA delta is too low and empty waiting queue; will densify singletons'
                )
                self.densify_singletons()
                self.should_run_densify = False

            elif (
                a is not None
                and self.params['min_delta_score_converge'] < a.delta_score()
            ):
                logger.info(f'ccPIC id is {self.ccpic_id}, Decision: augment graph from top LCA')
                self.queues.pop_Q()
                prs = a.get_inconsistent(
                    self.params['num_per_augmentation'], self.weight_mgr.futile_tester
                )
                if len(prs) == 0:
                    logger.info(f"ccPIC id is {self.ccpic_id},LCA marked as 'futile'. Moved to futile list.")
                    self.queues.add_to_futile(a)
                else:
                    self.weight_mgr.request_new_weights(prs)
                    self.queues.add_to_W(a)

            # Step 2e: at this point the only remaining active LCAs
            # are waiting for edges, so need to pause.
            elif self.queues.num_on_W() > 0:
                should_pause = True
                logger.info(f'ccPIC id is {self.ccpic_id},'
                    'Decision: top LCA delta is too low, but non-empty'
                    '  waiting queue, so need to pause'
                )

            # Step 2f: at this point, there are no active LCAs and no
            # LCAs are waiting for edges. The last possibility is to
            # densify the singleton LCAs. This is done at most once.
            # elif self.should_run_densify:
            #     logger.info(f'ccPIC id is {self.ccpic_id}, Decision: top LCA delta is too low and empty waiting queue; will densify singletons'
            #     )
            #     self.densify_singletons()
            #     self.should_run_densify = False

            # Step 2g: At this point, all active LCAs are waiting, and
            # if there are none then the algorithm has converged!
            else:
                assert self.queues.num_on_W() == 0
                logger.info(f'ccPIC id is {self.ccpic_id}, Decision: all deltas too low and empty waiting queue, so done'
                )
                converged = True

            #  Check
            if not converged and not should_pause:
                should_pause = self.check_wait_for_edges() or self.stop_request_check()
                if len(self.weight_mgr.waiting_for) == 0:
                    logger.info('Waiting for edges: <none>')
                else:
                    logger.info('Waiting for edges: %a' % self.weight_mgr.waiting_for)

            if should_pause and logger.getEffectiveLevel() <= logging.DEBUG:
                if len(self.weight_mgr.waiting_for) == 0:
                    logger.debug('Waiting for edges: <none>')
                else:
                    logger.debug('Waiting for edges: %a' % self.weight_mgr.waiting_for)

            if self.params['draw_iterations']:
                self.draw_obj.draw_iteration(
                    self.G, self.clustering, self.node2cid, iter_num
                )

            
            if self.trace_iter_compare_to_gt_cb is not None:
                self.trace_iter_compare_to_gt_cb(
                    self.clustering, self.node2cid, num_human, self.G
                )

            if (
                'max_human_decisions' in self.params
                and num_human > self.params['max_human_decisions']
            ):
                logger.info(
                    'Surpassed maximum number of human decisions with %d' % num_human
                )
                converged = True

        if self.progress_cb is not None:
            self.progress_cb(self, iter_num)

        logger.info(f'ccPIC id is {self.ccpic_id},'
            '*** Iteration %d Status Update - paused: %s, converged %s'
            % (
                iter_num,
                should_pause,
                converged,
            )
        )
        return (should_pause, iter_num, converged, halt_requested)

    def apply_lca(self, a):
        """
        Apply the LCA.  This involves (1) removing all other LCAs
        whose "from" set of clusters intersects a's "from" clusters,
        (2) forming new clusters, (3) generating new cluster
        singletons and/or pairs, and (4) forming new LCAs from them.
        """
        # Step 1: Get the cids of the clusters to be removed and get
        # the lcas to be removed
        old_cids = a.from_cids()
        old_lcas = self.cid2lca.remove_with_cids(old_cids)
        self.queues.remove(old_lcas)
        logger.info('Removing %d LCAs' % len(old_lcas))

        # Step 2: Form the new clusters and replace the old ones
        new_clusters = a.to_clusters.values()
        new_cids = self.generate_new_cids(len(new_clusters))
        added_clusters = {id: nodes for id, nodes in zip(new_cids, new_clusters)}
        ct.replace_clusters(old_cids, added_clusters, self.clustering, self.node2cid)

        #  Step 3: Form a list of CID singleton and/or pairs involving
        #  at least one of the new clusters.  Whether singletons or
        #  pairs or both included depends on the current phase of the
        #  computation.
        use_pairs, use_singles = self.which_lca_types()

        # Step 4: Generate the new LCAs and add them to the queue
        self.new_lcas(new_cids, use_pairs, use_singles)

    def compute_lca_scores(self):
        lcas_for_scoring = self.queues.get_S()
        for a in lcas_for_scoring:
            if self.phase == 'scoring':
                to_c, to_score = alg1.lca_alg1(a.subgraph)
            else:
                to_c, to_score = alg2.lca_alg2(
                    a.subgraph, a.from_cids(), a.from_node2cid()
                )
            a.set_to_clusters(to_c, to_score)
        self.queues.add_to_Q(lcas_for_scoring)
        self.queues.clear_S()

    def cids_for_edge(self, e):
        n0, n1, _ = e
        cid0 = self.node2cid[n0]
        cid1 = self.node2cid[n1]
        if cid0 == cid1:
            return [cid0]
        else:
            return [cid0, cid1]

    def add_edges(self):
        num_new_nodes = num_new_edges = 0
        new_cid_pairs = []
        for e in self.weight_mgr.get_weight_changes():
            if e[0] in self.removed_nodes or e[1] in self.removed_nodes:
                logger.info(f'ccPIC id is {self.ccpic_id}, Rejected edge {str(e)} because node was removed' )
                continue

            logger.info('Inserting edge %s' % str(e))

            #  For any new node, add it as an isolated cluster
            for node in (e[0], e[1]):
                if node not in self.node2cid:
                    new_cid = self.generate_new_cids()[0]
                    self.node2cid[node] = new_cid
                    self.clustering[new_cid] = set([node])
                    num_new_nodes += 1
                    logger.info(f'ccPIC id is {self.ccpic_id}, New node {node}, created new cid {new_cid}')

            cids = self.cids_for_edge(e)
            lcas_to_change = self.cid2lca.containing_all_cids(cids)

            # If no existing LCAs need to change and the edge joins
            # two clusters, then a new LCA is needed for the clusters.
            if len(lcas_to_change) == 0 and len(cids) == 2:
                cid_pair = (min(cids[0], cids[1]), max(cids[0], cids[1]))
                new_cid_pairs.append(cid_pair)

            #  Otherwise, adjust each LCA affected by the new edge.
            else:
                for a in lcas_to_change:
                    (from_delta, to_delta) = a.add_edge(e)
                    self.queues.score_change(a, from_delta, to_delta)

            # Incoporate the edge to the graph, depending on whether
            # on not it is a new connection.
            if e[1] in self.G[e[0]]:
                self.G[e[0]][e[1]]['weight'] += e[2]
            else:
                self.G.add_weighted_edges_from([e])
            num_new_edges += 1

            # The last step for this edge is to change the score
            if len(cids) == 1:
                self.score += e[2]
            else:
                self.score -= e[2]

        # At the end, generate new LCAs for newly connected pairs.
        if len(new_cid_pairs) > 0 and self.phase != 'splitting':
            self.create_lcas_from_tuples(new_cid_pairs)

        # return (num_new_nodes, num_new_edges)

    def remove_nodes(self):
        if self.remove_nodes_cb is None:
            return

        num_removed = 0
        for old_node in self.remove_nodes_cb(self.weight_mgr.node_set):
            if old_node not in self.G.nodes:
                continue

            logger.info(f'ccPIC id is {self.ccpic_id}, Removing node {old_node}' )

            self.removed_nodes.add(old_node)
            num_removed += 1

            # Get CID and set of nodes, removing the node
            old_cid = self.node2cid[old_node]
            del self.node2cid[old_node]
            old_cluster = self.clustering[old_cid]
            old_cluster.remove(old_node)

            # Remove LCAs
            lcas = self.cid2lca.remove_with_cids([old_cid])
            self.queues.remove(lcas)

            # Adjust the score based on removing the edges
            for n1 in self.G.neighbors(old_node):
                cid1 = self.node2cid[n1]
                if old_cid == cid1:
                    self.score -= self.G[old_node][n1]['weight']
                else:
                    self.score += self.G[old_node][n1]['weight']

            # Remove the node and its edges from the graph
            self.G.remove_node(old_node)

            # If the removed node was in a singleton cluster there is
            # nothing more to do.
            if len(old_cluster) == 0:
                del self.clustering[old_cid]
                logger.info(f"ccPIC id is {self.ccpic_id}, Removed node's cluster is now empty; no new LCAs")
                continue

            # Form the new clusters and replace the old one in the clustering
            subG = self.G.subgraph(old_cluster)
            cc = [comp for comp in nx.connected_components(subG)]
            new_cids = self.generate_new_cids(len(cc))
            new_clusters = {cid: c for cid, c in zip(new_cids, cc)}
            ct.replace_clusters([old_cid], new_clusters, self.clustering, self.node2cid)

            # Form the LCAs
            use_pairs, use_singles = self.which_lca_types()
            self.new_lcas(new_cids, use_pairs, use_singles)

    def densify_singletons(self):
        """
        Densify singleton LCAs, keeping a list of edges to add and
        placing LCAs adding at least one edge onto the waiting list.
        because they will wait for new edges to be weighted.
        All other LCAs are eliminated (though not the actual clusters,
        of course).  After this, the priority queue and scoring queuue
        should both be empty.  FIX ME
        """
        prs_to_add = []
        lcas = self.queues.Q.get_all().copy()
        self.queues.Q.clear()
        # self.cid2lca.clear()

        for a in lcas:
            if not a.is_singleton():  # need to write this method --- one liner
                self.queues.add_to_Q(a)
                continue
            to_add = a.densify_singleton(self.params)
            if len(to_add) > 0:
                self.queues.add_to_W(a)
                # self.cid2lca.add(a)
                prs_to_add.extend(to_add)
            else:
                self.queues.add_to_Q(a)
    
        if len(prs_to_add) > 0:
            self.weight_mgr.request_new_weights(prs_to_add)

    def status_check(self):
        active_scores = []
        for cid, lcas in self.cid2lca.items():
            scores = [a.delta_score() for a in lcas if a.to_clusters is not None]
            if len(scores) > 0:
                s = max(scores)
                if s > self.params.min_delta_score_converge:
                    active_scores.append(s)
        status_dict = {}
        a = len(active_scores)
        mx = max(active_scores)
        n = len(self.cid2lca)
        status_dict['max_score'] = mx
        status_dict['num_active'] = a
        status_dict['num_overall'] = n
        logger.info('Status: %d clusters out of %d are active' % (a, n))
        logger.info('Status: max LCA score is %d' % mx)
        return status_dict

    def provide_results(self):
        """
        Send back the current clusters
        """
        return self.clustering.values()

    def check_wait_for_edges(self):
        """
        Wait if there are enough edges waiting or if the waiting LCAs
        is high enough.
        """
        nw = self.queues.num_on_W()
        nl = self.queues.num_lcas()
        wait = nw > self.params['ga_max_num_waiting']
        if wait:
            logger.info(f'ccPIC id is {self.ccpic_id}, Decide to await for new edges: num waiting LCAs is {nw} out of {nl}')
        return wait

    def stop_request_check(self):
        return self.should_stop_cb is not None and self.should_stop_cb()

    def reset_waiting(self):
        self.weight_mgr.reset_waiting()
        self.queues.reset_waiting()
        logger.info(
            'Cleared waiting set. Scoring set now has %d LCAs' % len(self.queues.get_S)
        )

    def show_clustering(self):
        ct.print_structures(self.G, self.clustering, self.node2cid, self.score)

    def show_queues_debug(self):
        logger.debug('Scoring queue:')
        spaces = ' ' * 4
        if len(self.queues.S) == 0:
            logger.debug('  empty')
        else:
            for a in self.queues.S:
                a.pprint_short(initial_str=spaces, stop_after_from=True)

        logger.debug('Waiting queue:')
        if len(self.queues.W) == 0:
            logger.debug('  empty')
        else:
            for a in self.queues.W:
                a.pprint_short(initial_str=spaces, stop_after_from=True)

        logger.debug('Main queue:')
        if len(self.queues.Q.heap) == 0:
            logger.debug('  empty')
        else:
            for i, a in enumerate(self.queues.Q.heap):
                initial_str = '%3d:' % i
                a.pprint_short(initial_str=initial_str, stop_after_from=False)

    def show_brief_state(self):
        logger.info(f"ccPIC id is {self.ccpic_id}, LCAs {self.queues.num_lcas()}, clusters {len(self.clustering)}, new edges: {self.weight_mgr.edge_counts()} ")
            
        

        logger.info(
            'Queue lengths: main Q %d, scoring %d, waiting %d'
            % (len(self.queues.Q), len(self.queues.S), self.queues.num_on_W())
        )

        if len(self.queues.Q) == 0:
            logger.debug('Top LCA:  <none>')
        else:
            self.queues.top_Q().pprint_short(
                initial_str='Top LCA: ', stop_after_from=False
            )

    def is_consistent(self):
        """Each edge between two different clusters should be"""
        all_ok = self.queues.is_consistent()
        return all_ok  # Need to add a lot more here.....

# -*- coding: utf-8 -*-
"""
The interface to the database asks for properties like nodes, edges,
clusters and cluster ids. 

All edges are communicated as "quads" where each quad is of the form
  (n0, n1, w, aug_name)
Here, n0 and n1 are the nodes, w is the (signed!) weight and aug_name
is the augmentation method --- a verification algorithm or a human
annotator/reviewer -- that produced the edge.  Importantly, n0 < n1.

Nodes are generally added implicitly through adding edges, but as a
safety precaution all nodes in the clustering also added. This will
only add nodes that are disconnected from others in graph; such nodes
do not participate in the graph algorithm and are only recorded here
to avoid errors.
"""
import logging
import networkx as nx

import cluster_tools as ct


logger = logging.getLogger('lca')


class db_interface(object):  # NOQA
    def __init__(self, edges, clustering, are_edges_new=True):
        super(db_interface, self).__init__()

        self.edge_graph = nx.Graph()
        self.add_edges(edges, are_edges_new)
        self.clustering = clustering
        self.node_to_cid = ct.build_node_to_cluster_mapping(self.clustering)

        # Add any nodes that don't have edges.
        self.edge_graph.add_nodes_from(list(self.node_to_cid.keys()))

    def add_edges(self, quads, are_edges_new=True):
        """
        Add edges to the nx.Graph and, if the edges are new, to the
        underlying database.
        """
        self.edge_graph.add_edges_from([(n0, n1) for n0, n1, _, _ in quads])
        logger.info(f'Added {len(quads)} edges from quads')
        # num_components = len(list(nx.connected_components(self.edge_graph)))
        # logger.info(f'The current edge graph has {num_components} connected components')
        # logger.info(f'The current edge graph has {self.edge_graph.size()} edges')
        if are_edges_new:
            self.add_edges_db(quads)

    # def get_weight(self, triple):
    #     """
    #     Return the weight if the combination of n0, n1 and aug_name.
    #     If the aug_name is 'human' the summed weight is
    #     returned. Returns None if triple is unknown.
    #     """
    #     return self.get_weight_db(triple)

    def edges_from_attributes(self, n0, n1):
        try:
            return self.edges_from_attributes_db(n0, n1)
        except KeyError:
           return None

    def cluster_exists(self, cid):
        """
        Return True iff the cluster id exists in the clustering
        """
        return cid in self.clustering

    def get_cid(self, node):
        """
        Get the cluster id associated with a node. Returns None if
        cluster does not exist
        """
        try:
            return self.node_to_cid[node]
        except KeyError:
            return None

    def get_nodes_in_cluster(self, cid):
        """
        Find all the nodes the cluster referenced by cid.  Returns
        None if cluster does not exist.
        """
        try:
            return self.clustering[cid]
        except Exception:
            return None

    def edges_within_cluster(self, cid):
        """
        Find the multigraph edges that are within a cluster.
        Edges must be returned with n0<n1
        """
        quads = []
        cluster = sorted(self.clustering[cid])
        for i, ni in enumerate(cluster):
            for j in range(i + 1, len(cluster)):
                nj = cluster[j]
                if nj in self.edge_graph[ni]:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_leaving_cluster(self, cid):
        """
        Find the multigraph edges that connect between cluster cid and
        a different cluster.
        """
        quads = []
        cluster = self.clustering[cid]
        for ni in cluster:
            for nj in self.edge_graph[ni]:
                if nj not in cluster:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_between_clusters(self, cid0, cid1):
        """
        Find the multigraph edges that connect between cluster cid0
        and cluster cid1
        """
        assert cid0 != cid1
        quads = []
        cluster1 = self.clustering[cid1]
        for ni in self.clustering[cid0]:
            ei = self.edge_graph[ni]
            for nj in cluster1:
                if nj in ei:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_node_to_cluster(self, n, cid):
        """
        Find all edges between a node and a cluster.
        """
        raise NotImplementedError()

    def edges_between_nodes(self, node_set):
        """
        Find all edges between any pair of nodes in the node set.
        """
        quads = []
        node_list = sorted(node_set)
        for i, ni in enumerate(node_list):
            for j in range(i + 1, len(node_list)):
                nj = node_list[j]
                if nj in self.edge_graph[ni]:
                    quads.extend(self.edges_from_attributes(ni, nj))
        return quads

    def edges_from_node(self, node):
        quads = []
        for nj in self.edge_graph[node]:
            quads.extend(self.edges_from_attributes(node, nj))
        return quads

    def remove_nodes(self, nodes):
        for n in nodes:
            self.remove_node(n)

    def remove_node(self, n):
        if n not in self.edge_graph.nodes:
            return

        self.edge_graph.remove_node(n)
        cid = self.node_to_cid[n]
        del self.node_to_cid[n]
        self.clustering[cid].remove(n)
        if len(self.clustering[cid]) == 0:
            del self.clustering[cid]

    def commit_cluster_change(self, cc):
        """
        Commit the changes according to the type of change.  See
        compare_clusterings.py  Nothing is done here if there is
        no change, but there may need to be something done to the
        main WBIA database.
        """
        if cc.change_type != 'Unchanged':
            # 1. Add new clusters
            self.clustering.update(cc.new_clustering)

            # 2. Remove old clusters
            removed_cids = set(cc.old_clustering.keys()) - set(cc.new_clustering.keys())
            for old_c in removed_cids:
                del self.clustering[old_c]

            # 3. Update the node to clusterid mapping
            new_node_to_cid = ct.build_node_to_cluster_mapping(cc.new_clustering)
            self.node_to_cid.update(new_node_to_cid)

            # 4. Removed nodes should have already been removed from the
            #    db through the call to self.remove_nodes.
            for n in cc.removed_nodes:
                assert n not in self.node_to_cid

        return self.commit_cluster_change_db(cc)

    def add_edges_db(self, quads):
        raise NotImplementedError()

    # def get_weight_db(self, triple):
    #     raise NotImplementedError()

    def edges_from_attributes_db(self, n0, n1):
        raise NotImplementedError()

    def commit_cluster_change_db(self, cc):
        raise NotImplementedError()

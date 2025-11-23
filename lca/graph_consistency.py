
import random
import networkx as nx
import numpy as np
import logging
from tools import order_edge

# Update to graph_consistency.py

class GraphConsistencyAlgorithm(object):
    def __init__(self, config, classifier_manager, cluster_validator=None, G=nx.Graph()):
        """
        Args:
            classifier_manager: Handles single or multiple classifiers uniformly
            cluster_validator: Optional validation against ground truth
        """
        self.G = G.copy()
        self.config = config
        self.classifier_manager = classifier_manager
        self.cluster_validator = cluster_validator
        
        # Validation tracking
        self.validation_step = config.get('validation_step', 20)
        self.num_human_reviews = 0
        self.validation_initialized = False

        self.tries_before_edge_done = config.get('tries_before_edge_done', 4)
        self.human_attempts = {}

        print(f"self.THETA {self.config['theta']}")

    def filter_for_review(self, edges_for_human):
        logger = logging.getLogger('lca')
        final_for_review = []
        for n0, n1, s in edges_for_human:
            edge_key = order_edge(n0, n1)
            attempts = self.human_attempts.get(edge_key, 0)
            
            if attempts < self.tries_before_edge_done:
                self.human_attempts[edge_key] = attempts + 1
                final_for_review.append((n0, n1, s))
            else:
                logger.warning(f"Edge ({n0}, {n1}) exceeded max human attempts ({self.tries_before_edge_done})")
        return final_for_review

    def step(self, new_edges):
        """
        Perform one step of the graph consistency algorithm.
        Cycles through classifiers before requesting human review.
        """
        processed_edges = self.process_raw_edges(new_edges)
        self.add_new_edges(processed_edges)

        # Count human reviews for validation
        human_edges = [e for e in new_edges if 'human' in e[3]]
        self.num_human_reviews += len(human_edges)

        logger = logging.getLogger('lca')
        iPCCs = self.find_inconsistent_PCCs(logger)
        iPCCs = self.densify_iPCCs(iPCCs)
        
        edges_needing_classification = []
        for iPCC in iPCCs:
            logger.info(f"iPCC is {iPCC}")
            edges_needing_classification.extend(self.process_iPCC(iPCC))

        logger.info(f"{len(edges_needing_classification)} edges need classification")

        # Let ClassifierManager handle algorithmic classification vs human review decision
        new_classifications, edges_for_human = self.classifier_manager.classify_or_request_human(edges_needing_classification)
        
        # Add new algorithmic classifications to graph
        if new_classifications:
            logger.info(f"Can update {len(new_classifications)} edges")
            self.add_new_edges(new_classifications)
        edges_for_human = self.filter_for_review(edges_for_human)
        logger.info(f"{len(edges_for_human)} edges need human review")
        self.current_iPCCs = iPCCs
        
        # Validation logic
        self._handle_validation()

        return edges_for_human

    def _handle_validation(self):
        """Handle periodic validation against ground truth."""
        if not self.cluster_validator:
            return
            
        
        # Initial validation after first step
        if not self.validation_initialized:
            clustering, node2cid, G = self.get_clustering()
            self.cluster_validator.trace_start_human(clustering, node2cid, G, self.num_human_reviews)
            self.validation_initialized = True
            return
        
        # Periodic validation
        if hasattr(self.cluster_validator, 'prev_num_human'):
            if self.num_human_reviews - self.cluster_validator.prev_num_human >= self.validation_step:
                self.show_stats()
        else:
            # Fallback if prev_num_human not available
            if self.num_human_reviews % self.validation_step == 0:
                self.show_stats()
        
    def show_stats(self):
        clustering, node2cid, G = self.get_clustering()
        self.cluster_validator.trace_iter_compare_to_gt(clustering, node2cid, self.num_human_reviews, G)

    def densify_component(self, subG):
        max_edges = self.config["max_densify_edges"]
        logger = logging.getLogger('lca')
        
        missing_edges = []
        nodes = list(subG.nodes())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n0, n1 = nodes[i], nodes[j]
                if not self.G.has_edge(n0, n1):
                    missing_edges.append((n0, n1))
        if len(subG.edges()) + len(missing_edges) > max_edges:
            # Sample edges instead of full densification
            sample_size = max(0, max_edges - len(subG.edges()))
            logger.info(f"Sampling {sample_size}/{len(missing_edges)} edges for densification of CC with {len(subG.edges())} edges")
            missing_edges = random.sample(missing_edges, sample_size)
            
        # Classify missing edges using next available classifier for each
        new_edges = []
        for n0, n1 in missing_edges:
            edge = self.classifier_manager.classify_edge(n0, n1)
            new_edges.append(edge)

        self.add_new_edges(new_edges)
        return self.G.subgraph(subG.nodes())

    def densify_iPCCs(self, iPCCs):
        """Add missing edges using classifier system."""
        updated_iPCCs = []
        for subG in iPCCs:
            updated_iPCCs.append(self.densify_component(subG))
            
        return updated_iPCCs

    def deactivate(self, deactivator, deactivatee):

        n0, n1 = deactivator
        d0, d1 = deactivatee

        deativated_tuple = order_edge(d0, d1)

        self.G.edges[n0, n1]['inactivated_edges'].add(deativated_tuple)


        self.G.edges[d0, d1]['is_active'] = False
        self.G.edges[d0, d1]['deactivator'] = deactivator

    def is_finished(self):
        """
        Check if the algorithm is finished.
        
        Returns:
            bool: True if no more inconsistent PCCs exist
        """
        # Algorithm is finished when there are no inconsistent PCCs
        return len(getattr(self, 'current_iPCCs', [])) == 0

    def get_clustering(self):
        """
        Get current clustering results.
        Compatible with LCA interface.
        
        Returns:
            tuple: (clustering_dict, node2cid_dict)
        """
        return (*self.get_positive_clusters(), self.G)

    def get_positive_clusters(self):
        """
        Get clustering from the graph where clusters are connected components of positive edges.

        Args:
            G (nx.Graph): The graph containing nodes and edges with labels.

        Returns:
            cluster_dict (dict): A dictionary mapping cluster IDs to sets of nodes.
            node2cid (dict): A mapping of each node to its cluster ID.
        """

        positive_G = self.get_positive_subgraph(self.G)
        
        # Find connected components
        clusters = list(nx.connected_components(positive_G))
  
        used_nodes = {n for c in clusters for n in c}
        singletons = [n for n in self.G.nodes() if n not in used_nodes]#np.setdiff1d(list(self.G.nodes()), used_nodes)
        
        clusters = clusters + [{int(n)} for n in singletons]
        # print(clusters)

        # Convert list to a dictionary {cluster_id: set_of_nodes}
        cluster_dict = {cid: cluster for cid, cluster in enumerate(clusters)}
        
        # Create a node-to-cluster ID mapping
        node2cid = {node: cid for cid, cluster in cluster_dict.items() for node in cluster}

        return cluster_dict, node2cid

    def connect_iPCCs(self):
        logger = logging.getLogger('lca')
        cluster_dict, node2cid = self.get_positive_clusters()
        all_negative_edges = [(u, v, d["confidence"]) for u, v, d in self.G.edges(data=True) 
                          if d.get("label") == "negative" and
                          node2cid.get(u, -1) != node2cid.get(v, -2)]
        connections = {}
        for (u, v, d) in all_negative_edges:
            cluster_pair = order_edge(node2cid[u], node2cid[v])
            if cluster_pair not in connections:
                connections[cluster_pair] = []
            connections[cluster_pair].append((u, v, d))
        if not connections:
            return
        mean_confs = {pair:np.min([d for (_, _, d) in edges]) for (pair, edges) in connections.items()}
        mean_confs = {pair:conf for (pair, conf) in mean_confs.items() if conf < self.config["negative_threshold"]}
        # mean_confs = [(pair,np.median([d for (_, _, d) in edges])) for (pair, edges) in connections.items()]
        # mean_confs = sorted(mean_confs, key=lambda x: x[1])
        if not mean_confs:
            return
        max_pair = max(mean_confs, key=mean_confs.get)
        logger.info(f"Max pair: {max_pair}, {mean_confs[max_pair]}")
        if mean_confs[max_pair] < self.config["negative_threshold"]:
        # for (max_pair, mean_conf) in mean_confs.items():
            # if mean_conf > self.config["negative_threshold"]:
            #     break
            edges = [(u,v,c) for (u,v,c) in connections[max_pair] if not self.G[u][v]['auto_flipped']]
            if not edges:
                return
            u, v, c = max(edges, key=lambda x: x[2])
            u, v = order_edge(u, v)
            logger.info(f"Flipped edge {(u, v, c)} to connect clusters {cluster_dict[max_pair[0]]} and {cluster_dict[max_pair[1]]}")
            self.G[u][v]['label'] = "positive"
            self.G[u][v]['auto_flipped'] = True

    def process_raw_edges(self, raw_edges):
        """Convert raw edges to GC internal format using ClassifierManager."""
        processed_edges = []
        prob_human_correct = self.config.get('prob_human_correct', 0.98)
        
        for n0, n1, score, verifier_name in raw_edges:
            if verifier_name in {'human', 'simulated_human', 'ui_human'}:
                # Handle human decisions with config-based confidence
                decision = (score > 0.5)
                confidence = prob_human_correct
                label = "positive" if decision else "negative"
                processed_edges.append((n0, n1, score, confidence, label, verifier_name))
            else:
                # Use ClassifierManager to classify with appropriate classifier
                if verifier_name in self.classifier_manager.classifier_units:
                    embeddings, classifier = self.classifier_manager.classifier_units[verifier_name]
                    label, confidence = classifier.classify(score)
                    processed_edges.append((n0, n1, score, confidence, label, verifier_name))
        
        return processed_edges

    def add_new_edges(self, new_edges):
        """
        Add new edges from the ranker into the consistency graph. 
        Each edge is run through the verifier to generate the label and a confidence score before adding it to the graph.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
        """
        logger = logging.getLogger('lca')

        for n0, n1, score, confidence, label, ranker_name in new_edges:
        
            confidence = np.clip(confidence, 0, 1)
            # print(f"adding edge ... {n0}, {n1} {score}")
            # if label == "positive":
            #     positive_G = self.get_positive_subgraph(self.G)
            #     components = list(nx.connected_components(positive_G))
            #     c0 = next(filter(lambda c: n0 in c, components), {})
            #     c1 = next(filter(lambda c: n1 in c, components), {})
            #     if c0 != c1:
            #         logger.info(f"Added positive edge {n0, n1, float(score), float(confidence)} connecting clusters of size {max(len(c0), 1)} and {max(len(c1), 1)}")
            # else:
            #     logger.info(f"Negative confidence: {confidence}")
            if self.G.has_edge(n0, n1):
                old_edge = self.G.edges[n0, n1]
                if label != old_edge["label"] or confidence < old_edge["confidence"]:
                    for (v0, v1) in list(old_edge['inactivated_edges']):
                        self.G.edges[v0, v1]["is_active"] = True
                        self.G.edges[v0, v1]["deactivator"] = None
                    old_edge['inactivated_edges'] = set()
                    if not old_edge['is_active']:
                        old_edge['is_active'] = True
                        d0, d1 = old_edge["deactivator"]

                        self.G.edges[d0, d1]['inactivated_edges'].remove(order_edge(n0, n1))
                        old_edge["deactivator"] = None
                    
                
                logger.info(f"Updating existing edge ({n0}, {n1}) with label {label}, confidence {confidence}, score {score}, ranker {ranker_name}")
                self.G.edges[n0, n1]["label"] = label
                self.G.edges[n0, n1]["confidence"] = confidence
                self.G.edges[n0, n1]["ranker"] = ranker_name
                self.G.edges[n0, n1]["score"] = score
            else:
                self.G.add_edge(n0, n1, score=score, label=label, confidence=confidence, is_active=True, inactivated_edges=set(), deactivator=None, ranker=ranker_name, auto_flipped=False)
        

    def get_nonnegative_subgraph(self, G):
        # Extract positive edges
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") != "negative" and d.get('is_active')]

        # Create a subgraph with only positive edges
        positive_G = G.edge_subgraph(positive_edges).copy()
        singletons = [n for n in self.G.nodes() if n not in positive_G.nodes()]
        positive_G.add_nodes_from(singletons)

        return positive_G
    
    def get_positive_subgraph(self, G, min_confidence=0):
        # Extract positive edges
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == "positive" and d.get("confidence") > min_confidence and d.get('is_active')]

        # Create a subgraph with only positive edges
        positive_G = G.edge_subgraph(positive_edges).copy()
        singletons = [n for n in self.G.nodes() if n not in positive_G.nodes()]
        positive_G.add_nodes_from(singletons)
        return positive_G

    

    def find_inconsistent_PCCs(self, logger):
        """
        Find inconsistent Positive Connected Components, i.e. a connected components generate by positive edges, 
        with at least one negative edge between the nodes of that component.

        Returns a list of subgraphs corresponding to inconsistent Positive Connected Components.
        """
        result = []
        
        # Create a subgraph with only positive edges
        positive_G = self.get_positive_subgraph(self.G)
        
        # Find connected components
        components = list(nx.connected_components(positive_G))
        logger.info(f"Found connected components {len(components)}")   
        # logger.info(f"Total positive edges: {np.sum(d.get('label') == 'negative' for _, _, d in self.G.edges(data=True))}")
        # logger.info(f"Total negative edges: {np.sum(d.get('label') == 'positive' for _, _, d in self.G.edges(data=True))}")
        # Check each component for inconsistencies
        maxsize = 0
        for component in components:
            # logger.info(f"Nodes in component: {len(component)}")
            subG = self.G.subgraph(component)  # Get all edges within the component
            maxsize = max(maxsize, len(subG.nodes()))
            # subG = self.densify_component(subG)
            # if len(component) > self.config["densify_threshold"]:
            #     u, v, _ = min([(u,v, d["confidence"]) for u, v, d in subG.edges(data=True) if d["label"]=="positive"], key=lambda edge: edge[2])
            #     self.G[u][v]["label"] = "negative"
            # Check for negative edges
            if any(d.get("label") == "negative" and d.get("is_active") for _, _, d in subG.edges(data=True)):
                result.append(subG)
        logger.info(f"Largest component {maxsize}") 
        logger.info(f"Found inconsistent components {len(result)}")   
        return result


    def process_iPCC(self, iPCC):
        """
        Process an inconsistent PCC and return an edge that needs additional review.

        Args:
            iPCC (nx.Graph): a subgraph containing inconsistent PCC
        """
        result = set()
        
        # Find all negative edges in iPCC
        negative_edges = [(u, v) for u, v, d in iPCC.edges(data=True) if d.get("label") == "negative" and d.get('is_active')]
        # print(f"negative edges {negative_edges}")

        nonnegiPCC = self.get_nonnegative_subgraph(iPCC).copy()

        # cycles = list(nx.cycle_basis(iPCC))
        # print(f"Found {len(cycles)} cycles")
        auto_flip_edges = set()

        for n0, n1 in negative_edges:
            nonnegiPCC.add_edge(n0, n1, **iPCC[n0][n1])
            active_edges = [(u, v) for u, v, d in nonnegiPCC.edges(data=True) if d.get('is_active')]
            active_edges.append((n0, n1))
            neg_confidence = self.G[n0][n1]['confidence']

            activeiPCC = nonnegiPCC.edge_subgraph(active_edges).copy()
            cycles = list(nx.cycle_basis(activeiPCC))
            # Filter cycles that contain (n0, n1)
            valid_cycles = [cycle for cycle in cycles if n0 in cycle and n1 in cycle]
            # print(f"Found {len(valid_cycles)} valid cycles")
            
            if valid_cycles:
                # Find the cycle with the highest minimum confidence
                best_cycle = max(valid_cycles, key=lambda cycle: min(nonnegiPCC[u][v]['confidence'] for u, v in zip(cycle, cycle[1:] + [cycle[0]])))


                (u, v, min_confidence) = min(
                    [(u, v, nonnegiPCC[u][v]['confidence']) for u, v in zip(best_cycle, best_cycle[1:] + [best_cycle[0]]) if (u, v) != (n0, n1) and (v, u) != (n0, n1)],
                    key=lambda edge: edge[2]
                )
               
                if min_confidence > neg_confidence:
                    if abs(min_confidence - neg_confidence) > self.config['theta']:
                        self.deactivate((u,v), (n0, n1))
                    else:
                        result.add((*order_edge(n0, n1), self.G[n0][n1].get('ranker', None)))

                else:
                    if abs(min_confidence - neg_confidence) > self.config['theta']:
                        self.deactivate((n0, n1), (u,v))
                    else:
                        result.add((*order_edge(u, v), self.G[u][v].get('ranker', None)))
                
            nonnegiPCC.remove_edge(n0, n1)
        
        return result



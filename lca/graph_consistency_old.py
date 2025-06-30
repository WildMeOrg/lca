
import networkx as nx
import numpy as np
import logging

class GraphConsistencyAlgorithm(object):
    def __init__(self, config, classifier, G=nx.Graph()):
        """
        Initialize the graph consistency algorithm.

        Args:
            verifier (edge -> (confidence, "positive"|"negative"|"incomparable")): a 3-way verifier function that given an edge 
                                from a ranker classifies the edge as one of {"positive", "negative", "incomparable"} and 
                                assigns a confidence score in that classification.
            config : a dict containing configurable properties of the algorithm. Defaults to get_default_graph_consistency_config()
            G (nx.Graph, optional): the computation graph. Defaults to an empty nx.Graph().
        """
        self.G = G
        self.config = config
        self.classifier = classifier

    def deactivate(self, deactivator, deactivatee):

        n0, n1 = deactivator
        d0, d1 = deactivatee

        deativated_tuple = tuple(sorted([d0, d1]))

        self.G.edges[n0, n1]['inactivated_edges'].add(deativated_tuple)


        self.G.edges[d0, d1]['is_active'] = False
        self.G.edges[d0, d1]['deactivator'] = deactivator

        


    def step(self, new_edges):
        """
        Perform one step of the graph consistency algorithm.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
        """
        
        self.add_new_edges(new_edges)
        # self.process_human_decisions(human_decisions)

        logger = logging.getLogger('lca')
        # self.densify_PCCs(logger)
        # 
        # self.connect_PCCs()
        PCCs = self.find_inconsistent_PCCs(logger)
        
        PCCs = self.densify_PCCs(PCCs)
        
        for_review = []
        logger.info(f"Received {len(new_edges)} new edges")
        for PCC in PCCs:
            logger.info(f"PCC is {PCC}")
            for_review.extend(self.process_PCC(PCC))
            # print("extended graph for review")
        # print(f"{len(for_review)} edges to be reviewed")
        logger.info(f"{len(for_review)} edges to be reviewed")
        return PCCs, for_review

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

    def connect_PCCs(self):
        logger = logging.getLogger('lca')
        cluster_dict, node2cid = self.get_positive_clusters()
        all_negative_edges = [(u, v, d["confidence"]) for u, v, d in self.G.edges(data=True) 
                          if d.get("label") == "negative" and
                          node2cid.get(u, -1) != node2cid.get(v, -2)]
        connections = {}
        for (u, v, d) in all_negative_edges:
            cluster_pair = tuple(sorted([node2cid[u], node2cid[v]]))
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
            u, v = sorted([u, v])
            logger.info(f"Flipped edge {(u, v, c)} to connect clusters {cluster_dict[max_pair[0]]} and {cluster_dict[max_pair[1]]}")
            self.G[u][v]['label'] = "positive"
            self.G[u][v]['auto_flipped'] = True



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
            if label == "positive":
                positive_G = self.get_positive_subgraph(self.G)
                components = list(nx.connected_components(positive_G))
                c0 = next(filter(lambda c: n0 in c, components), {})
                c1 = next(filter(lambda c: n1 in c, components), {})
                if c0 != c1:
                    logger.info(f"Added positive edge {n0, n1, float(score), float(confidence)} connecting clusters of size {max(len(c0), 1)} and {max(len(c1), 1)}")
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

                        self.G.edges[d0, d1]['inactivated_edges'].remove(tuple(sorted([n0, n1])))
                        old_edge["deactivator"] = None
                    
                
                logger.info(f"Updating existing edge ({n0}, {n1})")
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

    def densify_PCCs(self, PCCs):
        """
        
        """
        
        logger = logging.getLogger('lca')
        
        updated_PCCs = []

        for subG in PCCs:
            # Find missing edges by checking all pair-wise connections
            missing_edges = []
            nodes = list(subG.nodes())
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    n0, n1 = nodes[i], nodes[j]
                    
                    if not self.G.has_edge(n0, n1):
                        missing_edges.append((n0, n1))
            logger.info(f"Found {len(missing_edges)} missing edges")

            # Add missing edges to the whole graph and then to this PCCs
            new_edges = [self.classifier(missing_edge) for missing_edge in missing_edges]
            self.add_new_edges(new_edges)
            updated_PCCs.append(self.G.subgraph(subG.nodes()))
        return updated_PCCs
            

        

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
        for component in components:
            # logger.info(f"Nodes in component: {len(component)}")
            subG = self.G.subgraph(component)  # Get all edges within the component
            
            # if len(component) > self.config["densify_threshold"]:
            #     u, v, _ = min([(u,v, d["confidence"]) for u, v, d in subG.edges(data=True) if d["label"]=="positive"], key=lambda edge: edge[2])
            #     self.G[u][v]["label"] = "negative"
            # Check for negative edges
            if any(d.get("label") == "negative" and d.get("is_active") for _, _, d in subG.edges(data=True)):
                result.append(subG)
        
        logger.info(f"Found inconsistent components {len(result)}")   
        return result


    def process_PCC(self, PCC):
        """
        Process an inconsistent PCC and return an edge that needs additional review.

        Args:
            PCC (nx.Graph): a subgraph containing inconsistent PCC
        """
        result = set()
        
        # Find all negative edges in PCC
        negative_edges = [(u, v) for u, v, d in PCC.edges(data=True) if d.get("label") == "negative" and d.get('is_active')]
        # print(f"negative edges {negative_edges}")

        nonnegPCC = self.get_nonnegative_subgraph(PCC).copy()

        # cycles = list(nx.cycle_basis(PCC))
        # print(f"Found {len(cycles)} cycles")
        auto_flip_edges = set()

        for n0, n1 in negative_edges:
            nonnegPCC.add_edge(n0, n1, **PCC[n0][n1])
            active_edges = [(u, v) for u, v, d in nonnegPCC.edges(data=True) if d.get('is_active')]
            active_edges.append((n0, n1))
            neg_confidence = self.G[n0][n1]['confidence']

            activePCC = nonnegPCC.edge_subgraph(active_edges).copy()
            cycles = list(nx.cycle_basis(activePCC))
            # Filter cycles that contain (n0, n1)
            valid_cycles = [cycle for cycle in cycles if n0 in cycle and n1 in cycle]
            # print(f"Found {len(valid_cycles)} valid cycles")
            
            if valid_cycles:
                # Find the cycle with the highest minimum confidence
                best_cycle = max(valid_cycles, key=lambda cycle: min(nonnegPCC[u][v]['confidence'] for u, v in zip(cycle, cycle[1:] + [cycle[0]])))


                (u, v, min_confidence) = min(
                    [(u, v, nonnegPCC[u][v]['confidence']) for u, v in zip(best_cycle, best_cycle[1:] + [best_cycle[0]]) if (u, v) != (n0, n1) and (v, u) != (n0, n1)],
                    key=lambda edge: edge[2]
                )
               
                if min_confidence > neg_confidence:
                    if abs(min_confidence - neg_confidence) > self.config['theta']:
                        self.deactivate((u,v), (n0, n1))
                    else:
                        result.add((n0, n1))

                else:
                    if abs(min_confidence - neg_confidence) > self.config['theta']:
                        self.deactivate((n0, n1), (u,v))
                    else:
                        result.add((u, v))
                
            nonnegPCC.remove_edge(n0, n1)
        
        return result

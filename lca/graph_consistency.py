
import networkx as nx
import numpy as np

class GraphConsistencyAlgorithm(object):
    def __init__(self, verifier, config, G=nx.Graph()):
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
        self.verifier = verifier
        self.config = config


    def step(self, new_edges, human_decisions, ranker_name):
        """
        Perform one step of the graph consistency algorithm.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
            human_decisions : list of human decisions of the form [(n0, n1, "positive"|"negative")]
        """
        
        self.add_new_edges(new_edges, ranker_name)
        self.process_human_decisions(human_decisions)

        PCCs = self.find_inconsistent_PCCs()

        for_review = []
        for PCC in PCCs:
            print(f"PCC is {PCC}")
            for_review.extend(self.process_PCC(PCC))
            print("extended graph for review")
        print(f"{len(for_review)} edges to be reviewed")
        return PCCs, for_review

    def add_new_edges(self, new_edges, ranker_name):
        """
        Add new edges from the ranker into the consistency graph. 
        Each edge is run through the verifier to generate the label and a confidence score before adding it to the graph.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
        """

        for n0, n1, score in new_edges:
            confidence, label = self.verifier((n0, n1, score, ranker_name))
        
            confidence = np.clip(confidence, 0, 1)
            # print(f"adding edge ... {n0}, {n1} {score}")
            self.G.add_edge(n0, n1, score=score, label=label, confidence=confidence, ranker=ranker_name, auto_flipped=False)
        

    def process_human_decisions(self, human_decisions):
        """
        Process human decisions and modify the graph accordingly.

        Args:
            human_decisions : list of human decisions of the form [(n0, n1, "positive"|"negative")]
        """
        # Go through each human decision and update the labels in the current graph, use the confidence from the config:
        for (n0, n1, human_label) in human_decisions:
          new_label = "positive" if human_label else "negative"
          self.G.edges[n0, n1]["label"] = new_label
          self.G.edges[n0, n1]["confidence"] = self.config["edge_weights"]["prob_human_correct"]
          self.G.edges[n0, n1]["ranker"] = "human"

    def get_nonnegative_subgraph(self, G):
        # Extract positive edges
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") != "negative"]

        # Create a subgraph with only positive edges
        positive_G = G.edge_subgraph(positive_edges)
        return positive_G
    
    def get_positive_subgraph(self, G):
        # Extract positive edges
        positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("label") == "positive"]

        # Create a subgraph with only positive edges
        positive_G = G.edge_subgraph(positive_edges)
        return positive_G

    def find_inconsistent_PCCs(self):
        """
        Find inconsistent Positive Connected Components, i.e. a connected components generate by positive edges, 
        with at least one negative edge between the nodes of that component.

        Returns a list of subgraphs corresponding to inconsistent Positive Connected Components.
        """
        result = []
        
        # Extract positive edges
        positive_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d.get("label") == "positive"]
        
        # Create a subgraph with only positive edges
        positive_G = self.get_positive_subgraph(self.G)
        
        # Find connected components
        components = nx.connected_components(positive_G)
        
        # Check each component for inconsistencies
        for component in components:
            subG = self.G.subgraph(component)  # Get all edges within the component
            
            # Check for negative edges
            if any(d.get("label") == "negative" for _, _, d in subG.edges(data=True)):
                result.append(subG)
        
        return result


    def process_PCC(self, PCC):
        """
        Process an inconsistent PCC and return an edge that needs additional review.

        Args:
            PCC (nx.Graph): a subgraph containing inconsistent PCC
        """
        result = set()
        
        # Find all negative edges in PCC
        negative_edges = [(u, v) for u, v, d in PCC.edges(data=True) if d.get("label") == "negative"]
        # print(f"negative edges {negative_edges}")

        nonnegPCC = self.get_nonnegative_subgraph(PCC).copy()

        # cycles = list(nx.cycle_basis(PCC))
        # print(f"Found {len(cycles)} cycles")
        auto_flip_edges = set()

        for n0, n1 in negative_edges:
            nonnegPCC.add_edge(n0, n1, **PCC[n0][n1])
            cycles = list(nx.cycle_basis(nonnegPCC))
            # Filter cycles that contain (n0, n1)
            valid_cycles = [cycle for cycle in cycles if n0 in cycle and n1 in cycle]
            # print(f"Found {len(valid_cycles)} valid cycles")
            
            if valid_cycles:
                # Find the cycle with the highest minimum confidence
                best_cycle = max(valid_cycles, key=lambda cycle: min(nonnegPCC[u][v]['confidence'] for u, v in zip(cycle, cycle[1:] + [cycle[0]])))

                (u, v, confidence) = min(
                    [(u, v, nonnegPCC[u][v]['confidence']) for u, v in zip(best_cycle, best_cycle[1:] + [best_cycle[0]])],
                    key=lambda edge: edge[2]
                )

                if confidence < self.config["flip_threshold"] and not self.G[u][v]['auto_flipped']:
                
                    auto_flip_edges.add((u, v))
                else:
                    result.add((u, v))
                
            nonnegPCC.remove_edge(n0, n1)
        
        for (u, v) in auto_flip_edges:
            self.G[u][v]['label'] = "positive" if self.G[u][v]['label'] == 'negative' else "negative"
            self.G[u][v]['confidence'] = 0.8 * self.G[u][v]['confidence']
            self.G[u][v]['auto_flipped'] = True

        return result



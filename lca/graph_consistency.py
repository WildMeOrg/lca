
import networkx as nx

def get_default_graph_consistency_config():
    return {"human_confidence":0.98}

class GraphConsistencyAlgorithm(object):
    def __init__(self, verifier, config=get_default_graph_consistency_config(), G=nx.Graph()):
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


    def step(self, new_edges, human_decisions):
        """
        Perform one step of the graph consistency algorithm.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
            human_decisions : list of human decisions of the form [(n0, n1, "positive"|"negative")]
        """
        
        self.add_new_edges(new_edges)
        self.process_human_decisions(human_decisions)

        PCCs = self.find_inconsistent_PCCs()

        for_review = []
        for PCC in PCCs:
            for_review.extend(self.process_PCC(PCC))

        return PCCs, for_review

    def add_new_edges(self, new_edges):
        """
        Add new edges from the ranker into the consistency graph. 
        Each edge is run through the verifier to generate the label and a confidence score before adding it to the graph.

        Args:
            new_edges : list of edges from the ranker of the form [(n0, n1, score, ranker_name),...]
        """

        for n0, n1, score, ranker_name in new_edges:
            confidence, label = self.verifier((n0, n1))
        
        confidence = max(0, min(1, confidence))
        self.G.add_edge(n0, n1, score=score, label=label, confidence=confidence, ranker=ranker_name)
        

    def process_human_decisions(self, human_decisions):
        """
        Process human decisions and modify the graph accordingly.

        Args:
            human_decisions : list of human decisions of the form [(n0, n1, "positive"|"negative")]
        """
        # Go through each human decision and update the labels in the current graph, use the confidence from the config:
        for (n0, n1, human_label) in human_decisions:
          self.G.edges[n0, n1]["label"] = human_label
          self.G.edges[n0, n1]["confidence"] = self.config["human_confidence"]     

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
        positive_G = self.G.edge_subgraph(positive_edges)
        
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
        result = []
        
        # Find all negative edges in PCC
        negative_edges = [(u, v) for u, v, d in PCC.edges(data=True) if d.get("label") == "negative"]
        
        for n0, n1 in negative_edges:
            # Find all cycles in PCC
            cycles = list(nx.simple_cycles(PCC.to_directed()))
            
            # Filter cycles that contain (n0, n1)
            valid_cycles = [cycle for cycle in cycles if n0 in cycle and n1 in cycle]
            
            if valid_cycles:
                # Find the cycle with the highest minimum confidence
                best_cycle = max(valid_cycles, key=lambda cycle: min(PCC[u][v]['confidence'] for u, v in zip(cycle, cycle[1:] + [cycle[0]])))
                
                # Identify the edge with the minimum confidence in the best cycle
                min_confidence_edge = min(
                    [(u, v) for u, v in zip(best_cycle, best_cycle[1:] + [best_cycle[0]])],
                    key=lambda edge: PCC[edge[0]][edge[1]]['confidence']
                )
                
                result.append(min_confidence_edge)
        
        return result



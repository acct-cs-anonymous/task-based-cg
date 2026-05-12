"""
Core equivalence classes learning and clustering logic.
"""
import pickle
import numpy as np
import networkx as nx
from init import ROOT_DIR


class EquivalenceClassLearner:
    """Handles loading and learning equivalence classes from composition metrics."""
    
    def __init__(self, K, threshold=0.01, final_output=True):
        if final_output:
            self.ce_metric_path = f"{ROOT_DIR}/equivalence_classes/ce_metric_dict_combination_{K}.pkl"
        else:
            self.ce_metric_path = f"{ROOT_DIR}/ce_metric_dict_combination_{K}_final_output_{final_output}.pkl"
        self.K = K
        self.threshold = threshold
        self.ce_metric = None
        self.unique_pairs = None
        self.ce_metric_dict_matrix = None
        self.clusters = None
        self.cluster_labels = None

    def load_ce_metric(self):
        """Load composition equivalence metric from pickle file."""
        with open(self.ce_metric_path, 'rb') as f:
            self.ce_metric = pickle.load(f)

    def convert_ce_metric_to_matrix(self):
        """Convert CE metric dictionary to matrix representation."""
        self.unique_pairs = set()
        for pair1, pair2 in self.ce_metric.keys():
            self.unique_pairs.add(pair1)
            self.unique_pairs.add(pair2)

        self.unique_pairs = list(self.unique_pairs)

        self.ce_metric_dict_matrix = np.zeros((len(self.unique_pairs), len(self.unique_pairs)))
        for i, pair1 in enumerate(self.unique_pairs):
            for j, pair2 in enumerate(self.unique_pairs):
                if pair1 == pair2:
                    self.ce_metric_dict_matrix[i, j] = 1.0
                else:
                    if (pair1, pair2) in self.ce_metric:
                        self.ce_metric_dict_matrix[i, j] = self.ce_metric[(pair1, pair2)]
                    else:
                        self.ce_metric_dict_matrix[i, j] = self.ce_metric[(pair2, pair1)]

    def cluster_ce_metric_matrix(self):
        """Cluster the CE metric matrix using graph-based connected components."""
        n = len(self.unique_pairs)
    
        # Create graph
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, label=self.unique_pairs[i])
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.ce_metric_dict_matrix[i, j] >= self.threshold:
                    G.add_edge(i, j, weight=self.ce_metric_dict_matrix[i, j])
        
        # Find clusters (connected components)
        self.clusters = list(nx.connected_components(G))
        self.cluster_labels = np.zeros(n, dtype=int)
        for cluster_id, cluster in enumerate(self.clusters):
            for node in cluster:
                self.cluster_labels[node] = cluster_id

    def get_combination_wise_acc(self, strategy, pos_embed):
        """Load and process combination-wise accuracy data."""
        acc_path = f"/{ROOT_DIR}/models/eval/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{self.K}/direct/train_{strategy}/eval_{strategy}/{pos_embed}/nh6_nl3/seed_0/accs.pkl"
        # acc_path = f"/{ROOT_DIR}/models/eval/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{self.K}/direct/train_{strategy}/eval_{strategy}/gemma1/accs.pkl"
        data_path = f"{ROOT_DIR}/data/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_{self.K}/direct/{strategy}/functions_info.pkl"
        
        with open(data_path, "rb") as f:
            functions_info = pickle.load(f)
        
        train_combination_accs = {}
        test_combination_accs = {}
        with open(acc_path, "rb") as f:
            accs = pickle.load(f)
            train_comb_acc = accs["train"].combination_sharp_acc
            test_comb_acc = accs["test"].combination_sharp_acc
            
        train_combinations = []
        test_combinations = []
        for cid, acc in test_comb_acc.items():
            function_id = [k for k, v in functions_info.items() if v == cid][0]
            test_combination_accs[function_id] = acc
            test_combinations.append(self.unique_pairs.index(function_id))
        
        for cid, acc in train_comb_acc.items():
            function_id = [k for k, v in functions_info.items() if v == cid][0]
            train_combination_accs[function_id] = acc
            train_combinations.append(self.unique_pairs.index(function_id))
        
        return train_combinations, test_combinations, train_combination_accs, test_combination_accs

    def print_cluster_info(self, skip_first_cluster=False):
        """Print information about discovered clusters."""
        print(f"\nFound {len(self.clusters)} clusters at threshold {self.threshold}:")
        print(self.clusters)
        print(self.cluster_labels)
        to_print_clusters = self.clusters

        for i, cluster in enumerate(to_print_clusters):
            members = [self.unique_pairs[n] for n in cluster]
            print(f"  Cluster {i} ({len(members)} nodes): {members}")
            
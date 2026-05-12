"""
Data splitting logic for train/test splits with equivalence class leakage.
"""
import math
import numpy as np
from sklearn.model_selection import train_test_split

class DataSplitter:
    """Handles splitting data into train/test sets with equivalence class control."""
    
    def __init__(self, unique_pairs, clusters, cluster_labels):
        self.unique_pairs = unique_pairs
        self.clusters = clusters
        self.cluster_labels = cluster_labels

    def create_disjoint_splits(self, seed=0, train_cluster_threshold=6, mode="even"):
        """Create disjoint splits."""
        np.random.seed(seed)
        train_task_indices = []
        test_task_indices = []
        train_clusters = []
        test_clusters = []
        remaining_clusters = []
        
        for idx, cluster in enumerate(self.clusters):
            # if length of cluster is less than 6, then add all elements to the training set
            if len(cluster) < train_cluster_threshold:
                train_task_indices.extend(cluster)
                train_clusters.append(cluster)
            else:
                remaining_clusters.append(cluster)

        # now randomly split remaining clusters into half size
        # sort remaining clusters by size in descending order
        if mode == "odd_reverse" or mode == "even_reverse" or mode == "odd_exchange_reverse" or mode == "even_exchange_reverse":
            remaining_clusters.sort(key=len, reverse=False)
        else:
            remaining_clusters.sort(key=len, reverse=True)
        self.train_test_cluster_pairs = []
        # assign first cluster to train and second cluster to test and so on
        print(f"Length of remaining clusters: {len(remaining_clusters)}")
        if len(remaining_clusters) % 2 == 1:
            max_index = len(remaining_clusters) - 1
        else:
            max_index = len(remaining_clusters)
        for i in range(0, max_index, 2):
            if mode in ["even", "even_reverse", "even_exchange", "even_exchange_reverse"]:
                self.train_test_cluster_pairs.append((remaining_clusters[i], remaining_clusters[i+1]))
            if mode in ["odd", "odd_reverse", "odd_exchange", "odd_exchange_reverse"]:
                self.train_test_cluster_pairs.append((remaining_clusters[i+1], remaining_clusters[i]))
            
            
        # add remaining cluster to tain if length of remaining cluster is odd
        if len(remaining_clusters) % 2 == 1:
            for n in remaining_clusters[-1]:
                train_task_indices.append(n)
            train_clusters.append(remaining_clusters[-1])
        for train_cluster, test_cluster in self.train_test_cluster_pairs:
            for n in train_cluster:
                train_task_indices.append(n)
            train_clusters.append(train_cluster)
            for n in test_cluster:
                test_task_indices.append(n)
            test_clusters.append(test_cluster)
        

        for train_cluster, test_cluster in self.train_test_cluster_pairs:
            print(f"Train cluster: {len(train_cluster)}")
            print(f"Test cluster: {len(test_cluster)}")
        print(f"Number of train clusters: {len(train_clusters)}")
        print(f"Number of test clusters: {len(test_clusters)}")
        print(f"Number of train task indices: {len(train_task_indices)}")
        print(f"Number of test task indices: {len(test_task_indices)}")
        
        return train_task_indices, test_task_indices

    def split_train_test_functions(self, train_task_indices, test_task_indices, seed=0, shared_equivalence_classes_percentage=0.5, mode="even"):
        """Create disjoint splits with leakage."""
        
        number_leaked_test_tasks = 0
        if shared_equivalence_classes_percentage > 0:
            train_task_indices, test_task_indices, number_leaked_test_tasks = self._leak_equivalence_classes_small_test_clusters(
                train_task_indices, test_task_indices, shared_equivalence_classes_percentage, mode=mode    
            )
        train_functions = [list(self.unique_pairs[n]) for n in train_task_indices]
        test_functions = [list(self.unique_pairs[n]) for n in test_task_indices]
        return train_functions, test_functions, train_task_indices, test_task_indices, number_leaked_test_tasks

    def _leak_equivalence_classes_small_test_clusters(self, train_task_indices, 
                                                      test_task_indices, 
                                                      shared_equivalence_classes_percentage,
                                                      mode="even"
                                                      ):
        """
        Leak equivalence classes by exchanging samples between train and test clusters in their pairs,
        using self.train_test_cluster_pairs.
        For each train-test cluster pair, exchange half of the minimum cluster size between train and test.
        """
        # Number of cluster pairs to leak
        num_pairs_to_leak = math.ceil(shared_equivalence_classes_percentage * len(self.train_test_cluster_pairs))
        number_leaked_test_tasks = 0

        print(f"Length of train_test_cluster_pairs: {len(self.train_test_cluster_pairs)}")
        print(f"train_test_cluster_pairs: {self.train_test_cluster_pairs}")
        print(f"Number of cluster pairs to leak: {num_pairs_to_leak}")

        number_leaked_test_tasks = 0
        for idx in range(num_pairs_to_leak):
            if idx >= len(self.train_test_cluster_pairs):
                break

            train_cluster = list(self.train_test_cluster_pairs[idx][0])
            test_cluster = list(self.train_test_cluster_pairs[idx][1])

            if "exchange" in mode:
                min_cluster_len = min(len(train_cluster), len(test_cluster))
            else:
                min_cluster_len = len(test_cluster)
            if min_cluster_len < 1:
                continue
            half_min = min_cluster_len // 2
            test_to_leak = test_cluster[:half_min]
            for n in test_to_leak:
                if n in test_task_indices:
                    test_task_indices.remove(n)
                    train_task_indices.append(n)
            number_leaked_test_tasks += len(test_cluster) - len(test_to_leak)
                
            if mode in ["even_exchange", "even_exchange_reverse", "odd_exchange", "odd_exchange_reverse"]:
            # # Leak half_min from train -> test
                train_to_leak = train_cluster[:half_min]
                for n in train_to_leak:
                    if n in train_task_indices:
                        train_task_indices.remove(n)
                        test_task_indices.append(n)
            
                number_leaked_test_tasks += len(train_to_leak)

        return train_task_indices, test_task_indices, number_leaked_test_tasks

    
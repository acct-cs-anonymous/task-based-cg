"""
Main orchestrator class that combines all components.
"""
import os
from init import ROOT_DIR
from src.data.equivalence_classes.learn_equivalence_classes import EquivalenceClassLearner
from src.data.equivalence_classes.data_splitter import DataSplitter
from src.data.equivalence_classes.visualizer import EquivalenceClassVisualizer
from matplotlib import pyplot as plt


class CompositionEquivalenceClasses:
    """
    Main class for managing composition equivalence classes.
    Orchestrates learning, splitting, and visualization.
    """
    
    def __init__(self, K, threshold=0.01, final_output=True):
        self.K = K
        self.threshold = threshold
        self.learner = EquivalenceClassLearner(K, threshold, final_output)
        self.splitter = None
        self.visualizer = None

    def load_ce_metric(self):
        """Load composition equivalence metric."""
        self.learner.load_ce_metric()

    def convert_ce_metric_to_matrix(self):
        """Convert CE metric to matrix representation."""
        self.learner.convert_ce_metric_to_matrix()

    def cluster_ce_metric_matrix(self):
        """Cluster the CE metric matrix."""
        self.learner.cluster_ce_metric_matrix()
        
        # Initialize splitter and visualizer after clustering
        self.splitter = DataSplitter(
            self.learner.unique_pairs,
            self.learner.clusters,
            self.learner.cluster_labels
        )
        self.visualizer = EquivalenceClassVisualizer(
            self.K,
            self.learner.unique_pairs,
            self.learner.ce_metric_dict_matrix,
            self.learner.cluster_labels
        )

    def split_train_test_functions(self, train_task_indices, test_task_indices, seed=0, shared_equivalence_classes_percentage=0, visualize_split=False, skip_small_clusters=False, cluster_size_threshold=6, mode="even"):
        """
        Split functions into train and test sets.
        
        Args:
            shared_equivalence_classes_percentage: Fraction of equivalence classes to share
            train_split_percentage: Fraction of data for training
        
        Returns:
            train_functions: List of training function pairs
            test_functions: List of test function pairs
        """
        if self.splitter is None:
            raise RuntimeError("Must call cluster_ce_metric_matrix() first")
        
        train_functions, test_functions, train_task_indices, test_task_indices, number_leaked_test_tasks = \
            self.splitter.split_train_test_functions(
                train_task_indices,
                test_task_indices,
                seed,
                shared_equivalence_classes_percentage,
                mode=mode
            )

        # print(f"Number of train tasks: {len(train_task_indices)}")
        # print(f"Number of test tasks: {len(test_task_indices)}")
        leaked_test_tasks_percentage = number_leaked_test_tasks / len(test_task_indices)
        print(f"Number of leaked test tasks: {number_leaked_test_tasks}/{len(test_task_indices)} = {leaked_test_tasks_percentage}")

        # Visualize the split
        if visualize_split:
            shared_equivalence_classes_fig = self.visualizer.visualize_clustered_ce_metric_matrix(
                highlight_train_indices=train_task_indices, 
                highlight_test_indices=test_task_indices,
                shared_equivalence_classes_percentage=shared_equivalence_classes_percentage,
                skip_small_clusters=skip_small_clusters,
                cluster_size_threshold=cluster_size_threshold
            )
            
            # Save visualization
            plots_dir = f"{ROOT_DIR}/equivalence_classes/plots/disjoint_splits/{self.K}_{shared_equivalence_classes_percentage}"
            os.makedirs(plots_dir, exist_ok=True)
            base_filename = (f"clustered_ce_metric_matrix_train_test_highlight_shared_"
                            f"{self.K}_{shared_equivalence_classes_percentage}")
            shared_equivalence_classes_fig.savefig(f"{plots_dir}/{base_filename}.pdf")
            shared_equivalence_classes_fig.savefig(f"{plots_dir}/{base_filename}.png")

        return train_functions, test_functions, number_leaked_test_tasks

    def visualize_train_test_combination_accs(self, shared_equivalence_classes_percentage, 
                                             pos_embed="abs", plot_dir=None, 
                                             cluster_size_min_threshold=6, cluster_size_max_threshold=100, plot_type="matrix", strategy=None):
        """
        Visualize train/test combination accuracies.
        
        Args:
            shared_equivalence_classes_percentage: Fraction of shared classes
            pos_embed: Position embedding type
            plot_dir: Directory to save plots
        
        Returns:
            Matplotlib figure object
        """
        if self.visualizer is None:
            raise RuntimeError("Must call cluster_ce_metric_matrix() first")
        
        strategy = f"disjoint2_{self.K}_{int(shared_equivalence_classes_percentage * 100)}"
        # strategy = "combination_6"
        train_combination_indices, test_combination_indices, train_combination_accs, test_combination_accs = \
            self.learner.get_combination_wise_acc(strategy, pos_embed)
        
        skip_first_cluster = (self.K == 6)
        
        if plot_type == "matrix":
            acc_fig = self.visualizer.visualize_clustered_ce_metric_matrix(
                highlight_train_indices=train_combination_indices,
                highlight_test_indices=test_combination_indices,
                pos_embed=pos_embed,
                test_combination_accs=test_combination_accs,
                cluster_size_min_threshold=cluster_size_min_threshold,
                cluster_size_max_threshold=cluster_size_max_threshold,
                shared_equivalence_classes_percentage=shared_equivalence_classes_percentage
            )
            base_filename = f"clustered_ce_metric_matrix_train_test_highlight_shared_{self.K}_{shared_equivalence_classes_percentage}"
        elif plot_type == "bar":
            acc_fig = self.visualizer.visualize_distribution_plot_accuracy_vs_shared_equivalence_classes(
                train_combination_accs=train_combination_accs,
                test_combination_accs=test_combination_accs,
                shared_equivalence_classes_percentage=shared_equivalence_classes_percentage,
                strategy=strategy
            )
            base_filename = f"distribution_plot_accuracy_vs_shared_equivalence_classes_{self.K}_{shared_equivalence_classes_percentage}"
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")
        
        # Save plots
        if plot_dir:
            acc_fig.savefig(f"{plot_dir}/{base_filename}.png")
            acc_fig.savefig(f"{plot_dir}/{base_filename}.pdf")

        return acc_fig

    def visualize_clustered_ce_metric_matrix(self, **kwargs):
        """Delegate to visualizer for direct visualization access."""
        if self.visualizer is None:
            raise RuntimeError("Must call cluster_ce_metric_matrix() first")
        return self.visualizer.visualize_clustered_ce_metric_matrix(**kwargs)

    def print_cluster_info(self, skip_first_cluster=False):
        """Print information about discovered clusters."""
        self.learner.print_cluster_info(skip_first_cluster)


def main():
    """Example usage of the CompositionEquivalenceClasses system."""
    for K in [6]:
        for shared_equivalence_classes_percentage in [0.0, 0.25, 0.5, 0.75, 1.0]:
            threshold = 0.01
            # Initialize and load data
            composition_equivalence_classes = CompositionEquivalenceClasses(K, threshold)
            composition_equivalence_classes.load_ce_metric()
            composition_equivalence_classes.convert_ce_metric_to_matrix()
            composition_equivalence_classes.cluster_ce_metric_matrix()
            composition_equivalence_classes.print_cluster_info()
            # print number of clusters and their sizes
            
            
            plot_dir = f"{ROOT_DIR}/equivalence_classes/plots/combination_accs/{K}_{shared_equivalence_classes_percentage}"
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            strategy = f"disjoint2_{K}_{int(shared_equivalence_classes_percentage * 100)}"
            composition_equivalence_classes.visualize_train_test_combination_accs(shared_equivalence_classes_percentage=shared_equivalence_classes_percentage, pos_embed="rel_global", plot_dir=plot_dir, cluster_size_min_threshold=23, cluster_size_max_threshold=100, plot_type="matrix")
            composition_equivalence_classes.visualize_train_test_combination_accs(shared_equivalence_classes_percentage=shared_equivalence_classes_percentage, pos_embed="abs", plot_dir=plot_dir, cluster_size_min_threshold=0, cluster_size_max_threshold=600, plot_type="bar", strategy=strategy)
if __name__ == "__main__":
    main()  
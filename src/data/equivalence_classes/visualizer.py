"""
Visualization utilities for composition equivalence classes.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
import json

class EquivalenceClassVisualizer:
    """Handles all visualization for equivalence classes."""
    
    def __init__(self, K, unique_pairs, ce_metric_dict_matrix, cluster_labels):
        self.K = K
        self.unique_pairs = unique_pairs
        self.ce_metric_dict_matrix = ce_metric_dict_matrix
        self.cluster_labels = cluster_labels

    def visualize_clustered_ce_metric_matrix(self, highlight_train_indices=None, 
                                            highlight_test_indices=None, 
                                            pos_embed="abs", 
                                            test_combination_accs=None, 
                                            shared_equivalence_classes_percentage=0,
                                            cluster_size_min_threshold=6,
                                            cluster_size_max_threshold=100):
        """
        Visualize the clustered composition equivalence matrix.
        
        Args:
            highlight_train_indices: Indices to highlight as training data
            highlight_test_indices: Indices to highlight as test data
            pos_embed: Position embedding type
            test_combination_accs: Dictionary of test accuracies
            skip_first_cluster: Whether to skip largest clusters in visualization
            shared_equivalence_classes_percentage: Percentage of shared classes
        
        Returns:
            Matplotlib figure object
        """
        # get cluster labels of test combination accuracies
        print(cluster_size_min_threshold, cluster_size_max_threshold)
        font_size = 60
        plt.rcParams.update({
            'font.size': font_size,
            'legend.fontsize': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size
        })
        
        if self.K == 6:
            fig, ax = plt.subplots(figsize=(80, 80))
        else:
            fig, ax = plt.subplots(figsize=(50, 50))
        
        # Organize clusters by label
        cluster_to_indices = defaultdict(list)
        for idx, label in enumerate(self.cluster_labels):
            cluster_to_indices[label].append(idx)

        test_flags = [0] * len(self.cluster_labels)
        selected_labels = []
        labels = [self.unique_pairs[i] for i in range(len(self.cluster_labels))]
        for idx, entry in enumerate(labels):
            if entry in test_combination_accs:
                selected_labels.append(self.cluster_labels[idx])
                test_flags[idx] = 1
        # Determine which clusters to plot
        clusters_to_plot = self._get_clusters_to_plot(
            cluster_to_indices, cluster_size_min_threshold, cluster_size_max_threshold, selected_labels
        )

        
        sorted_idx = self._get_sorted_indices(clusters_to_plot, test_flags)
        
        # Prepare sorted data
        sorted_matrix = self.ce_metric_dict_matrix[sorted_idx][:, sorted_idx]
        sorted_labels = [self.unique_pairs[i] for i in sorted_idx]
        sorted_cluster_ids = [self.cluster_labels[i] for i in sorted_idx]
        sorted_cluster_sizes = [len(cluster_to_indices[self.cluster_labels[i]]) for i in sorted_idx]
        

        # Plot matrix
        im = ax.imshow(sorted_matrix, cmap='Blues', vmin=0, vmax=1, alpha=0.3)

        # annotate cluster number per boundary
        prev_label = None
        for idx in range(len(sorted_idx)):
            if self.cluster_labels[sorted_idx[idx]] != prev_label:
                ax.text(idx + 20, idx + 10, f"Class ID {self.cluster_labels[sorted_idx[idx]]}", fontweight='bold', fontsize=font_size-10, ha='center', va='center')
                prev_label = self.cluster_labels[sorted_idx[idx]]

        # Configure axes
        self._configure_axes(ax, sorted_labels, font_size)
        
        # Add title
        title = (f'Composition Equivalence Score Matrix, K = {self.K}')
        ax.set_title(title, fontsize=font_size, fontweight='bold', pad=30)
        
        # Add colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        # Increase the pad value to add a larger gap between the matrix and the colorbar
        cax = divider.append_axes("right", size="2.5%", pad=3.0)  # Increased pad for a visible gap
        cbar = plt.colorbar(im, cax=cax, label='Composition Equivalence Score')
        cbar.ax.tick_params(labelsize=font_size-10)
        # cbar.ax.set_title('Composition Equivalence Score', fontsize=font_size-10)
        
        # Draw cluster boundaries
        self._draw_cluster_boundaries(ax, sorted_idx)
        
        # Highlight train/test indices
        self._add_train_test_highlights(ax, sorted_labels, 
                                       highlight_train_indices, 
                                       highlight_test_indices)
        
        # Add accuracy annotations
        if test_combination_accs is not None:
            self._add_accuracy_annotations(ax, sorted_labels, test_combination_accs, sorted_cluster_sizes, sorted_idx, font_size)
        
        # Add legend
        self._add_legend(ax)
        
        plt.tight_layout()
        
        return fig

    def _get_clusters_to_plot(self, cluster_to_indices, cluster_size_min_threshold=6, cluster_size_max_threshold=100, selected_labels=None):
        """Determine which clusters to include in visualization."""

        return {
            lbl: idxs for lbl, idxs in cluster_to_indices.items() 
            if len(idxs) >= cluster_size_min_threshold and len(idxs) <= cluster_size_max_threshold and lbl in selected_labels
        }
        

    def _get_sorted_indices(self, clusters_to_plot, test_flags):
        """Sort indices by cluster size (descending) and cluster label (ascending)."""
        decorated = []
        for cluster_label, indices in clusters_to_plot.items():
            cluster_size = len(indices)
            for idx in indices:
                decorated.append((cluster_size, cluster_label, test_flags[idx], idx))

        decorated_sorted = sorted(decorated, key=lambda x: (-x[0], x[1], x[2]))
        return [x[3] for x in decorated_sorted]

    def _configure_axes(self, ax, sorted_labels, font_size):
        """Configure axis ticks and labels."""
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
        ax.set_xticks(np.arange(len(sorted_labels)))
        ax.set_yticks(np.arange(len(sorted_labels)))
        ax.set_xticklabels(sorted_labels, rotation=90, ha='left', 
                          fontsize=font_size - 20, fontweight='normal')
        ax.set_yticklabels(sorted_labels, fontsize=font_size - 20, fontweight='normal')
        
        ax.tick_params(axis='x', which='major', pad=15)
        ax.tick_params(axis='y', which='major', pad=10)

    def _draw_cluster_boundaries(self, ax, sorted_idx):
        """Draw lines between different clusters."""
        cluster_sorted = [self.cluster_labels[x] for x in sorted_idx]
        
        for i in range(1, len(sorted_idx)):
            if cluster_sorted[i] != cluster_sorted[i-1]:
                ax.axhline(i - 0.5, color='gray', linewidth=2)
                ax.axvline(i - 0.5, color='gray', linewidth=2)

    def _add_train_test_highlights(self, ax, sorted_labels, 
                                   highlight_train_indices, 
                                   highlight_test_indices):
        """Add markers for train and test data points."""
        if highlight_train_indices is not None:
            for i in highlight_train_indices:
                entry = self.unique_pairs[i]
                if entry not in sorted_labels:
                    continue
                index = sorted_labels.index(entry)
                label = 'Train' if i == 0 else None
                ax.scatter(index, index, color='green', s=1200, marker='s', 
                         linewidths=3, label=label)
                
        if highlight_test_indices is not None:
            for i in highlight_test_indices:
                entry = self.unique_pairs[i]
                if entry not in sorted_labels:
                    continue
                index = sorted_labels.index(entry)
                label = 'Test' if i == 0 else None
                ax.scatter(index, index, color='red', s=1200, marker='s', 
                         linewidths=3, label=label)

    def _add_accuracy_annotations(self, ax, sorted_labels, test_combination_accs, sorted_cluster_sizes, sorted_idx, font_size):
        """Add accuracy values as vertical bars at class bottom edges."""
        for k, v in test_combination_accs.items():
            print(k, v.item())
        
        # Draw vertical bars at the bottom edge of each equivalence class
        bar_width = 0.3  # Width of the accuracy bar
        bar_color = 'red'
        bar_alpha = 0.8
        
        
        prev_class_size = None
        class_size = None
        flip_sign = False
        cumulative_size = sorted_cluster_sizes[0]
        for i, entry in enumerate(sorted_labels):
            if entry in test_combination_accs:
                print(i, entry, sorted_cluster_sizes[i], cumulative_size)
                acc_value = test_combination_accs[entry]
                
                # Get the size of this equivalence class
                class_size = sorted_cluster_sizes[i]
                if prev_class_size is None:
                    prev_class_size = class_size
                if prev_class_size != class_size:
                    # cumulative_size += prev_class_size
                    flip_sign = True
                    prev_class_size = class_size
                
                # Draw vertical bar at the bottom edge of this class
                if flip_sign:
                    # Draw upwards (positive height)
                    ax.bar(
                        x=i,
                        height=-acc_value*10,  # Positive height
                        bottom=cumulative_size - 1,  # Start from cumulative_size
                        color=bar_color,
                        alpha=bar_alpha,
                        edgecolor='black',
                        linewidth=0.5
                    )
                else:
                    # Draw downwards (negative height, but starting at cumulative_size)
                    ax.bar(
                        x=i,
                        height=acc_value*10,  # Negative height, so bar goes downward
                        bottom=cumulative_size,
                        color=bar_color,
                        alpha=bar_alpha,
                        edgecolor='black',
                        linewidth=0.5
                    )
                
                # Optional: Add text label showing accuracy value
                if not flip_sign:
                    y_axis = cumulative_size + acc_value*10 + .05
                else:
                    y_axis = cumulative_size - acc_value*10 - .05 - 3
                ax.text(
                    x=i,
                    s=f"{acc_value:.2f}",
                    y = y_axis,  # Slightly above the bar
                    fontsize=font_size - 10,
                    ha='center',
                    va='top',
                    rotation=90
                )
            # add cumulatve size only when cluster changes
            
    def _add_legend(self, ax):
        """Add legend with train and test markers."""
        handles, labels = ax.get_legend_handles_labels()
        legend_entries = {l: h for h, l in zip(handles, labels)}

        # Ensure both train and test appear in legend
        if 'Train' not in legend_entries:
            legend_entries['Train'] = Line2D(
                [0], [0], marker='s', color='w', markerfacecolor='green', 
                markersize=18, linewidth=0, label='Train'
            )
        if 'Test' not in legend_entries:
            legend_entries['Test'] = Line2D(
                [0], [0], marker='s', color='w', markerfacecolor='red', 
                markersize=18, linewidth=0, label='Test'
            )
            
        ax.legend(list(legend_entries.values()), list(legend_entries.keys()),
                 loc='lower left', markerscale=2, fontsize=45, frameon=True)

    def visualize_distribution_plot_accuracy_vs_shared_equivalence_classes(self, 
                                            train_combination_accs=None,
                                            test_combination_accs=None, 
                                            shared_equivalence_classes_percentage=0,
                                            cluster_size_min_threshold=6,
                                            strategy=None):
        # calculate mean and std accuracy per test cluster
        # set font size to 10
        font_size = 20
        plt.rcParams.update({
            'font.size': font_size,
            'legend.fontsize': font_size,
            'axes.labelsize': font_size,
            'axes.titlesize': font_size
        })
        cluster_to_function_ids = defaultdict(list)
        cluster_to_shared_map = {}
        cluster_size = {}
        unique_cluster_ids = list(set(self.cluster_labels))
        # set cluster_to_shared_map to False for all cluster ids
        for cluster_id in unique_cluster_ids:
            cluster_to_shared_map[cluster_id] = False
            cluster_size[cluster_id] = 0
        for idx, cluster_id in enumerate(self.cluster_labels):
            cluster_size[cluster_id] += 1
            if self.unique_pairs[idx] in test_combination_accs:
                cluster_to_function_ids[cluster_id].append(test_combination_accs[self.unique_pairs[idx]])
            if self.unique_pairs[idx] in train_combination_accs:
                cluster_to_shared_map[cluster_id] = True

        json_cluster_to_function_ids = {str(k): v for k, v in cluster_to_function_ids.items()}
        json_cluster_to_shared_map = {str(k): v for k, v in cluster_to_shared_map.items()}
        json_cluster_size = {str(k): v for k, v in cluster_size.items()}
        json_dict = {"cluster_to_function_ids": json_cluster_to_function_ids, "cluster_to_shared_map": json_cluster_to_shared_map, "cluster_size": json_cluster_size}
        # save cluster_to_function_ids to a json file
        with open(f'{strategy}_json_dict.json', 'w') as f:
            json.dump(json_dict, f)

        # mean and std accuracy per test cluster
        mean_accuracy_per_test_cluster = {}
        std_accuracy_per_test_cluster = {}
        for cluster_id, function_ids in cluster_to_function_ids.items():
            mean_accuracy_per_test_cluster[cluster_id] = np.mean(function_ids)
            std_accuracy_per_test_cluster[cluster_id] = np.std(function_ids)
            print(cluster_id, cluster_size[cluster_id], cluster_to_shared_map[cluster_id], mean_accuracy_per_test_cluster[cluster_id], std_accuracy_per_test_cluster[cluster_id])
            

        # plot the error bar plot of mean accuracy and color by shared map
        
        fig, ax = plt.subplots(figsize=(10, 10))
        for cluster_id, mean_accuracy, in mean_accuracy_per_test_cluster.items():
            # have a circle where size is cluster_size[cluster_id]
            ax.scatter(cluster_id, mean_accuracy, s=cluster_size[cluster_id]*10, color='green' if cluster_to_shared_map[cluster_id] else 'red')
            # add error bars
            ax.errorbar(cluster_id, mean_accuracy, yerr=std_accuracy_per_test_cluster[cluster_id], color='black', linestyle='None')
            
        # Create handles for shared/non-shared legend (color)
        color_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=12, linewidth=0, label='Shared'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=12, linewidth=0, label='Not Shared')
        ]

        # include only cluster sizes that are in mean_accuracy_per_test_cluster
        new_cluster_size = {}
        for cluster_id, size in cluster_size.items():
            if cluster_id in mean_accuracy_per_test_cluster:
                new_cluster_size[cluster_id] = size
        cluster_size = new_cluster_size
        # Create handles for circle size legend
        import matplotlib.patches as mpatches
        max_cluster = max(cluster_size.values())
        min_cluster = min(cluster_size.values())
        # Pick three representative sizes: small, medium, large
        size_legend_vals = [
            min_cluster,
            (min_cluster + max_cluster) // 2,
            max_cluster
        ]
        size_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=np.sqrt(val*10), linewidth=0, label=f'Size={val}')
            for val in size_legend_vals
        ]

        # Add both legends inside the plot, on the top right
        legend1 = ax.legend(
            handles=color_handles, 
            loc='upper right', 
            bbox_to_anchor=(1, 1),  
            markerscale=1.5, 
            fontsize=font_size, 
            frameon=True, 
            ncol=1
        )
        ax.add_artist(legend1)
        legend2 = ax.legend(
            handles=size_handles, 
            loc='upper right', 
            bbox_to_anchor=(1, 0.78),  # below 'Shared Class' legend, still inside plot
            title='Class Size', 
            fontsize=font_size-2, 
            frameon=True, 
            ncol=1
        )
        plt.subplots_adjust(bottom=0.33)  # make room for legends below plot
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        
        ax.set_xlabel('Equivalence Class ID', fontsize=font_size)
        ax.set_ylabel('Accuracy', fontsize=font_size)
        ax.set_title('Accuracy distribution for test sequences\nwith and without shared equivalence classes', fontsize=font_size)
        plt.tight_layout()
        return fig
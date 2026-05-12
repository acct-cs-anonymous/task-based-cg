import numpy as np
from init import read_config, ROOT_DIR
import pickle
import argparse
from src.data.composition_generator.compositions import CompositionsGenerator
import os

class CEMetricCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.function_type == "diverse":
            self.max_seq_len = 2 * self.cfg.seq_len
        else:
            self.max_seq_len = self.cfg.seq_len

    def sample_string(self):
        """
        Samples a string of the given length.
        """
        n_alphabets = self.cfg.n_alphabets
        seq_len = self.cfg.seq_len
        with_replacement = self.cfg.with_replacement
        alph = [chr(i + 97) for i in range(n_alphabets)]
        tokens = np.random.choice(alph, size=seq_len, replace=with_replacement)
        return "".join(tokens)

    def calculate_ce_metric(self, all_functions, composition_generator, N_samples=1000, test_functions=None):
        if test_functions is None:
            test_functions = all_functions
        ce_metric = 0
        # Complexity of the algorithm is O(N * M * L), 
        # where N is the number of train functions, 
        # M is the number of test functions, 
        # and N_samples is the number of samples to calculate the CE metric * number of functions in each composition.
        ce_metric_dict = {}
        # Vectorized version for faster computation
        n_functions = len(all_functions)
        # Each row: a function composition
        # We'll precompute N_samples random input pairs for all function pairs
        # Prepare random input strings for all function pairs and all samples
        input_strings_1 = [self.sample_string() for _ in range(N_samples)]
        input_strings_2 = [self.sample_string() for _ in range(N_samples)]

        # Precompute all outputs for all functions on all N_samples input pairs
        all_outputs = dict()
        for idx, func in enumerate(all_functions):
            outputs = []
            for s1, s2 in zip(input_strings_1, input_strings_2):
                out = composition_generator.apply_function_composition(func, s1, s2)[-1][:self.max_seq_len]
                outputs.append(out)
            all_outputs[idx] = np.array(outputs, dtype=object)  # array of strings

        ce_metric = 0
        ce_metric_dict = {}
        # Compare all upper-triangular pairs only (i<j)
        n_pairs = n_functions * (n_functions - 1) // 2
        for i in range(n_functions):
            for j in range(i + 1, n_functions):
                # Compare the outputs over all N_samples
                matches = all_outputs[i] == all_outputs[j]
                pairwise_ce = np.sum(matches) / N_samples
                pair = (tuple(all_functions[i]), tuple(test_functions[j]))
                ce_metric_dict[pair] = ce_metric_dict.get(pair, 0) + pairwise_ce
                ce_metric += pairwise_ce

        ce_metric = ce_metric / n_pairs
        return ce_metric, ce_metric_dict

    def _calculate_ce_metric_for_single_function(self, train_function, test_function, composition_generator, N_samples=10000):
        ce_metric = 0
        for i in range(N_samples):
            input_string_1, input_string_2 = self.sample_string(), self.sample_string()
            train_output = composition_generator.apply_function_composition(train_function, input_string_1, input_string_2)[-1][:self.max_seq_len]
            test_output = composition_generator.apply_function_composition(test_function, input_string_1, input_string_2)[-1][:self.max_seq_len]
         
            ce_metric += train_output == test_output
        return ce_metric / N_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--K",
        type=int,
        default=6,
        help="size of the composition"
    )
    args = parser.parse_args()
    cfg = read_config("./config/gen/conf.yaml")
    ce_metric_calculation = CEMetricCalculator(cfg)
    cfg.split_strategy = f"combination_{args.K}"
    cfg.task_max_length = args.K
    cfg.function_type = "diverse"
    compositions_generator = CompositionsGenerator(cfg)
    train_functions, test_functions, functions_info = compositions_generator.get_train_test_compositions()
        
    all_functions = train_functions + test_functions
    print(f"All functions: {len(all_functions)}")
    
    ce_metric, ce_metric_dict = ce_metric_calculation.calculate_ce_metric(all_functions, compositions_generator, N_samples=1000)
    print(f"CE metric: {ce_metric}")
    
    # save ce_metric_dict in data directory
    dir_path = f"{ROOT_DIR}/equivalence_classes"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(f"{dir_path}/ce_metric_dict_{cfg.split_strategy}.pkl", "wb") as f:
        pickle.dump(ce_metric_dict, f)
    
if __name__ == "__main__":
    main()
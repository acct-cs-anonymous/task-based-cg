#!/bin/bash

# Generate data for within-k evaluation without identity functions (as task max length=k) (Figure 1)
for k in {2..6}; do
    echo "Generating data for combination_$k"
    python -m scripts.generate_data  --split_strategy combination_$k --task_max_length $k --functions_type uniform
    python -m scripts.generate_data  --split_strategy combination_$k --task_max_length $k --functions_type diverse
done


# Generate data for within-k and cross-k evaluation with identity functions as task max length=7 (Figure 2)
for k in {1..6}; do
    echo "Generating data for combination_$k"
    python -m scripts.generate_data  --split_strategy combination_$k --task_max_length 7 --functions_type uniform
    python -m scripts.generate_data  --split_strategy combination_$k --task_max_length 7 --functions_type diverse
done

# Generate data for controlled splits after learning equivalence classes in diverse benchmark for K=6 (leakage with swapping test members to keep constant test size)
for percentage in 0 10 20 50 60 70 100; do
    python -m scripts.generate_data --split_strategy disjoint7_6_$percentage --task_max_length 6 --function_type diverse # (Figure 3)
done

# Generate data for controlled splits after learning equivalence classes in diverse benchmark for K=6 (leakage without swapping test members)
for percentage in 0 10 20 50 60 70 100; do
    python -m scripts.generate_data --split_strategy disjoint9_6_$percentage --task_max_length 6 --function_type diverse # (Figure 13)
done


# Generate data for module coverage based position-wise and pairwise divergence for task max length=6 (Figure 4 and 19)

for percentage in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python -m scripts.generate_data --split_strategy continuouscoverage_6_$percentage --task_max_length 6 --function_type diverse
    python -m scripts.generate_data --split_strategy continuouscoverage_6_$percentage --task_max_length 6 --function_type uniform
    python -m scripts.generate_data --split_strategy continuouspaircoverage_6_$percentage --task_max_length 6 --function_type diverse
    python -m scripts.generate_data --split_strategy continuouspaircoverage_6_$percentage --task_max_length 6 --function_type uniform
done


# generate data to validate composition equivalence necessity using uniform-identity based equivalences (Figure 10a)
# We focus on K=6, keep train perm size fixed as 576, test perm size fixed as 144 and systematically increase number of shared compositions in train and test
shared_equivalence_size=(0 1 25 49 73 97 121 144)
for size in "${shared_equivalence_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy equivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
done

# generate data to validate number of training equivalences needed to learn corresponding test equivalences (Figure 10b)
# We focus on K=6, keep train perm size fixed as 576, and shared equivalences as 100% and systematically increase number of training equivalences
training_equivalence_size=(0)
for size in "${training_equivalence_size[@]}"; do
    python -m scripts.generate_data --prompt_length fixed --split_strategy uniequivalence_6_576_$size --n_functions 6 --task_max_length 7 --n_alphabets 26 --seq_len 6 --functions_type uniform 
done


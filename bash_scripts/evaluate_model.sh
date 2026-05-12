#!/bin/bash
N_ALPHABETS=26
SEQ_LEN=6
N_FUNCTIONS=6
NHEADS_NLAYERS="nh6_nl3"


# (within-k evaluation without identity functions as task max length=k)
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --train_split "combination_3" \
    --eval_split "combination_3" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "abs" \
    --function_type "diverse" \
    --task_max_length 3 \
    --seed 0 

# (within-k evaluation with identity functions as task max length=7)
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --train_split "combination_3" \
    --eval_split "combination_3" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "abs" \
    --function_type "diverse" \
    --task_max_length 7 \
    --seed 0 

# (cross-k evaluation with identity functions as task max length=7)
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --train_split "combination_3" \
    --eval_split "combination_6" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "abs" \
    --function_type "diverse" \
    --task_max_length 7 \
    --seed 0
    

# (controlled splits evaluation of Gemma3-1B model on diverse benchmark for K=6 (Figure 3))
python -m scripts.evaluate_model \
    --prompt_mode "direct" \
    --train_split "disjoint7_6_60" \
    --eval_split "disjoint7_6_60" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "abs" \
    --function_type "diverse" \
    --task_max_length 6 \
    --seed 0 \
    --pretrained True

# (module coverage based position-wise divergence evaluation of Nanogpt step-by-step mode on diverse benchmark for K=6 (Figure 4))
python -m scripts.evaluate_model \
    --prompt_mode "step_by_step" \
    --train_split "continuouscoverage_6_0.0" \
    --eval_split "continuouscoverage_6_0.0" \
    --nheads_nlayers "$NHEADS_NLAYERS" \
    --pos_embedding_type "abs" \
    --function_type "diverse" \
    --task_max_length 6 \
    --seed 0 

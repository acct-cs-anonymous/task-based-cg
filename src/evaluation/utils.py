import numpy as np
import pickle
import logging
import torch

def map_docs_to_combination_id(docs, token_manager, token_map=None):
    """Map documents to their combination IDs"""
    docs_functions = np.array([token_manager.get_function_list(doc, token_map) for doc in docs])

    combination_ids_map = {}
    combination_ids = []

    # vectorized version of the loop
    if token_map is not None:
        docs_function_token = [
            tuple([token_manager.token[token_map[fn]] for fn in docs_functions[i]])
            for i in range(len(docs_functions))
        ]
    else:
        docs_function_token = [
            tuple([token_manager.token[fn] for fn in docs_functions[i]])
            for i in range(len(docs_functions))
        ]
    # convert to list of tuples

    combination_ids = np.array(
        [
            token_manager.functions_info[docs_function_token[i]]
            for i in range(len(docs_function_token))
        ]
    )
    combination_ids_map = {
        cid: docs_function_token[i] for i, cid in enumerate(combination_ids)
    }
    return combination_ids, docs_function_token


def calculate_combination_accuracy(
    acc_array, ood_flags, combination_ids
):
    """Calculate accuracy grouped by combination ID"""
    print_error_indices = []

    combination_acc = {}
    combination_mean_acc = {}
    combination_ood = {}
    combination_indices = {}

    for idx, combination_id in enumerate(combination_ids):
        if combination_id not in combination_acc:
            combination_acc[combination_id] = []
            combination_mean_acc[combination_id] = []
            combination_ood[combination_id] = []
            combination_indices[combination_id] = []
        
        sharp_acc_val = acc_array[idx].all(dim=-1).float().mean().item()
        mean_acc_val = acc_array[idx].float().mean(dim=-1).item()
        ood_val = ood_flags[idx]

        combination_acc[combination_id].append(sharp_acc_val)
        combination_mean_acc[combination_id].append(mean_acc_val)
        combination_ood[combination_id].append(ood_val)
        combination_indices[combination_id].append(idx)
    # find total number of unique combination ids
    total_unique_combination_ids = len(set(combination_ids))
    # per unique combination id, print 10% of the indices where the accuracy is 0
    for cid in combination_acc:

        if len(combination_acc[cid]) > 0 and cid not in [189, 153]:
            # find top 10 indices where the accuracy is 0
            K = min(10, len(combination_acc[cid]))
            top_K_indices = np.argsort(combination_acc[cid])[:K]
            top_K_indices = [combination_indices[cid][i] for i in top_K_indices]
            print_error_indices.extend(top_K_indices)
        else:
            # print all indices
            print_error_indices = combination_indices[cid]
    print_error_indices = list(set(print_error_indices))
    return (
        {cid: np.mean(accs) for cid, accs in combination_acc.items()},
        {cid: np.mean(means) for cid, means in combination_mean_acc.items()},
        {cid: np.mean(oods) for cid, oods in combination_ood.items()},
        total_unique_combination_ids,
        print_error_indices,
    )


def get_print_indices(combination_ids):
    """Get indices for printing examples"""
    print_indices = []
    if combination_ids is not None:
        unique_ids = list(np.unique(np.array(combination_ids)))
        for id in unique_ids:
            indices = np.where(np.array(combination_ids) == id)
            print_indices.append(indices[0][-1])
    return print_indices


def is_ood_prompt_vectorized(
    token, token_idx, output_batch, target_output_batch, prompt_length
):
    """Vectorized OOD detection keeping original logic"""
    task_tokens = ["sort", "map", "filter", "union", "join", "max", "identity"]
    if prompt_length == "variable":
        structure_tokens = ["<SEP>", "<END>", " "]
    else:
        structure_tokens = ["<SEP>", "<END>"]

    # Convert to indices for vectorized operations
    task_indices = torch.tensor(
        [token_idx[t] for t in task_tokens if t in token_idx],
        dtype=output_batch.dtype,
        device=output_batch.device,
    )
    structure_indices = torch.tensor(
        [token_idx[t] for t in structure_tokens if t in token_idx],
        dtype=output_batch.dtype,
        device=output_batch.device,
    )

    # Check task tokens in output (vectorized is_function_in_output)
    has_task_tokens = torch.isin(output_batch, task_indices).any(dim=1)

    # Check structure mismatch (vectorized is_structure_mismatch)
    structure_mask = torch.isin(output_batch, structure_indices)
    structure_mismatch = structure_mask & (output_batch != target_output_batch)
    has_structure_mismatch = structure_mismatch.any(dim=1)

    # Print OOD examples for debugging
    ood_flags = has_task_tokens | has_structure_mismatch
    if ood_flags.any():
        print("\n=== {} OOD Examples Found ===".format(ood_flags.sum()))
        # print the first 5 ood examples
        MAX_PRINT = min(5, len(ood_flags))
        for i in range(MAX_PRINT):
            if ood_flags[i]:
                print(f"Sample {i} (OOD):")
                output_tokens = [token[t.item()] for t in output_batch[i]]
                target_tokens = [token[t.item()] for t in target_output_batch[i]]
                print(f"  Output: {output_tokens}")
                print(f"  Target: {target_tokens}")
                print(f"  Has task tokens: {has_task_tokens[i].item()}")
                print(f"  Has structure mismatch: {has_structure_mismatch[i].item()}")
                print()

    return ood_flags


def is_ood_prompt(token, token_idx, output_batch, target_output_batch, prompt_length):
    """Optimized batch OOD detection"""
    return torch.tensor(
        is_ood_prompt_vectorized(
            token, token_idx, output_batch, target_output_batch, prompt_length
        )
    )


def load_train_test_accs(accs_path, functions_info_path):
    with open(accs_path, "rb") as f:
        train_test_accs = pickle.load(f)

    with open(functions_info_path, "rb") as f:
        functions_info = pickle.load(f)

    ck, accs = train_test_accs[0]

    test_comb_acc = accs["test"]["combination"]["acc"]
    train_comb_acc = accs["train"]["combination"]["acc"]
    train_acc = accs["train"]["total"]["acc"]
    test_acc = accs["test"]["total"]["acc"]

    test_combination_accs = {}
    train_combination_accs = {}
    for cid, acc in test_comb_acc.items():
        # find corresponding function names
        function_id = [k for k, v in functions_info.items() if v == cid][0]
        test_combination_accs[function_id] = acc

    for cid, acc in train_comb_acc.items():
        # find corresponding function names
        function_id = [k for k, v in functions_info.items() if v == cid][0]
        train_combination_accs[function_id] = acc

    return train_acc, test_acc, train_combination_accs, test_combination_accs



def log_verbose_results(
    logger, split, docs, doc_combination_id_map, metrics, ck, prompt_mode
):
    """Log verbose results for standard evaluation"""
    logger.info("Evaluating {} documents: {}".format(split, len(docs)))
    combination_acc = metrics["combination"]["acc"]
    combination_ood = metrics["combination"]["ood"]
    total_acc = metrics["total"]["acc"]
    total_ood = metrics["total"]["ood"]
    if prompt_mode == "step_by_step":
        module_wise_acc = metrics["module_wise"]
        step_by_step_acc = metrics["step_by_step"]
        direct_acc = metrics["direct"]

    for cid, acc in combination_acc.items():
        logger.info(
            "Accuracy for combination id {} {} is {:.3f}".format(
                cid, doc_combination_id_map[cid], acc
            )
        )

    for cid, ood in combination_ood.items():
        logger.info(
            "OOD for combination id {} {} is {:.3f}".format(
                cid, doc_combination_id_map[cid], ood
            )
        )

    logger.info(
        "Iter: {} Split: {} Acc: {:.3f} OOD: {:.3f}".format(
            ck, split, total_acc, total_ood
        )
    )
    if prompt_mode == "step_by_step":
        for k, v in module_wise_acc.items():
            logger.info("Module wise acc for {} is {:.3f}".format(k, np.mean(v["acc"])))
        for k, v in step_by_step_acc.items():
            logger.info("Step by step acc for {} is {:.3f}".format(k, np.mean(v)))
        for k, v in direct_acc.items():
            logger.info("Direct acc for {} is {:.3f}".format(k, np.mean(v["acc"])))


# Additional utility function for batch processing
def batch_process_functions(
    dat_batch, token_idx, sep_token, decode, get_function_list, batch_size=None
):
    """Process function extraction for entire batch at once"""
    if batch_size is None:
        batch_size = dat_batch.shape[0]

    dat_np = dat_batch.cpu().numpy()
    sep_token_idx = token_idx[sep_token]

    # Vectorized function list extraction
    all_function_lists = []
    all_decoded_lists = []

    for i in range(batch_size):
        function_list = get_function_list(dat_np[i], sep_token_idx)
        decoded_list = decode(function_list, return_list=True)
        all_function_lists.append(function_list)
        all_decoded_lists.append(decoded_list)

    return all_function_lists, all_decoded_lists

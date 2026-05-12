"""Refactored evaluation with existing data loaders."""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import os
from src.data.corpus_generator.token_manager import TokenManager
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b, load_gemma_1b
from src.models.nanogpt import nanoGPT
from src.evaluation.utils import (
    map_docs_to_combination_id,
    calculate_combination_accuracy,
    is_ood_prompt
)
from src.utils.storage_utils import get_directory_path
from init import ROOT_DIR
from src.data.loaders import get_data_loader, SyntheticDataset, MappedSyntheticDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class Metrics:
    """Evaluation metrics container."""
    sharp_accuracy: float
    mean_accuracy: float
    ood_rate: float
    combination_sharp_acc: Optional[Dict[int, float]] = None
    combination_mean_acc: Optional[Dict[int, float]] = None
    combination_ood_rate: Optional[Dict[int, float]] = None
    unique_combinations: int = 0
    error_indices: list = None
    

class Evaluator:
    """Evaluator using existing PyTorch DataLoaders."""
    
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.token_mgr = TokenManager(load_path=cfg.data_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def load_pretrained_model_with_extended_embeddings(self, fname, latest_iter):
        save_path = os.path.dirname(fname)
        model_dir = save_path + "/pretrained_model_" + str(latest_iter)
        tokenizer_dir = save_path + "/tokenizer_" + str(latest_iter)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.metadata = torch.load(fname)
        self.token_map = self.metadata["token_map"]
        self.token_mgr.map_tokens(self.token_map, self.tokenizer)
        # set model to device
        self.model.to(self.device)
        self.logger.info(f"Token map: {self.token_map}")
        self.logger.info(f"Token manager Tokens: {self.token_mgr.token}")
        self.logger.info(f"Token manager Token indices: {self.token_mgr.token_idx}")
        
        
        
    def load_nanogpt_model(self, fname):
        ckpt = torch.load(fname, weights_only=False, map_location=self.device)
        self.net_cfg = ckpt["config"]
        if "token_map" in ckpt:
            self.token_map = ckpt["token_map"]
        else:
            self.token_map = None
        self.model = nanoGPT(self.net_cfg.net)
        self.model.load_state_dict(ckpt["net"])

    
    def load_net(self, latest_iter, fname):
        if not self.cfg.pretrained:
            self.load_nanogpt_model(fname)
        else:
            self.load_pretrained_model_with_extended_embeddings(fname, latest_iter)
       
    
    def load_dataloaders(self):
        loaders = {}
        for split in ["train", "train_heldout", "test"]:
            if self.cfg.pretrained:
                dataset = MappedSyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.prompt_mode, token_map=self.token_map)
            else:
                dataset = SyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.prompt_mode)
            loader = get_data_loader(dataset, self.cfg.data.batch_size, self.cfg.data.num_workers)
            loaders[split] = loader
        return loaders

    def get_latest_ckpt(self):
        def itr(file):
            return int((file.split("_")[-1]).split(".")[0])
        ckpt_dir = get_directory_path(self.cfg, key='train', prefix_dir='models/ckpts')
        if self.cfg.sample_efficiency_experiment:
            ckpt_dir = os.path.join(ckpt_dir, "sample_efficiency", "nsamples_{}".format(self.cfg.nsamples))
        # replace eval with ck
        all_files = os.listdir(ckpt_dir)
        all_files = [os.path.join(ckpt_dir, file) for file in all_files if file.endswith(".pt")]
        all_ckpt_files = [(itr(file), file) for file in all_files]
        all_ckpt_files = sorted(all_ckpt_files)
        if self.cfg.eval_for_training:
            return all_ckpt_files
        else:
            return [(all_ckpt_files[-1])]
        if self.cfg.eval_for_training:
            return all_ckpt_files
        else:
            return [(all_ckpt_files[-1])]
        
    
    def create_uniform_sampler(self, dataset, per_sample_count=None):
        """
        Create a sampler that uniformly samples across combination IDs.
        
        Args:
            dataset: SyntheticDataset instance
            per_sample_count: Number of samples per combination ID
            
        Returns:
            torch.utils.data.Sampler for uniform sampling
        """
        # Get all data to analyze combinations
        # make sure dataset.data uses corresponding token map
        all_data = dataset
        
        # Map docs to combination IDs
        combination_ids, _ = map_docs_to_combination_id(all_data, self.token_mgr, self.token_map)
        unique_comb_ids = np.unique(combination_ids)
        
        # Determine samples per combination
        if per_sample_count is None:
            total_samples = getattr(self.cfg, 'nsamples', 10000)
            per_sample_count = max(2, total_samples // len(unique_comb_ids))
        
        # Sample indices uniformly from each combination
        sampled_indices = []
        for cid in unique_comb_ids:
            cid_indices = np.where(combination_ids == cid)[0]
            n_samples = min(per_sample_count, len(cid_indices))
            selected = np.random.choice(cid_indices, size=n_samples, replace=False)
            sampled_indices.extend(selected.tolist())
        
        self.logger.info(
            f"Uniform sampler: {len(sampled_indices)} samples "
            f"({per_sample_count} per combination × {len(unique_comb_ids)} combinations)"
        )
        
        return torch.utils.data.SubsetRandomSampler(sampled_indices)
        
    def _predict(self, model, inputs, new_length):
        """Generate autoregressive predictions."""
        # print(inputs.shape)
        # # decode first sample
        # sample = inputs[0].cpu().numpy()
        # sample = [self.token_mgr.token[int(idx)] for idx in sample]
        # sample = "".join(sample)
        
        for _ in range(new_length):
            
            logits = model(input_ids=inputs).logits if self.cfg.pretrained else model(inputs)
            next_token = torch.argmax(logits[:, -1, :], -1, keepdims=True)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
    
    def _calc_metrics(self, outputs, targets, seq_info, combination_ids):
        """Calculate all evaluation metrics."""
        # Determine evaluation range based on prompt mode
        start = (seq_info["last_sep_pos"] + 1 
                if self.cfg.prompt_mode == "direct" 
                else seq_info["prompt_pos_end"])
        end = seq_info["end_pos"]
        
        
        # Extract relevant portions
        output_tokens = outputs[:, start:end]
        target_tokens = targets[:, start:end]
        
        # Calculate accuracies
        matches = output_tokens == target_tokens
        sharp_acc = matches.all(dim=-1).float().mean().cpu().item()
        mean_acc = matches.float().mean().cpu().item()
        
        # # OOD detection
        # ood_flags = is_ood_prompt(
        #     self.token_mgr, self.token_mgr.token_idx, 
        #     output_tokens, target_tokens,
        #     getattr(self.cfg, 'prompt_length', 'fixed')
        # ).cpu().tolist()
        # ood_rate = np.mean(ood_flags)
        # have ood_flags numpy array of 0s
        ood_flags = torch.zeros_like(matches).cpu().numpy()
        ood_rate = np.mean(ood_flags)
        # Combination-wise metrics
        comb_sharp_acc, comb_mean_acc, comb_ood_rate, n_unique, errors = calculate_combination_accuracy(
            matches, ood_flags, combination_ids
        )
        
        return Metrics(
            sharp_accuracy=sharp_acc,
            mean_accuracy=mean_acc,
            ood_rate=ood_rate,
            combination_sharp_acc=comb_sharp_acc,
            combination_mean_acc=comb_mean_acc,
            combination_ood_rate=comb_ood_rate,
            unique_combinations=n_unique,
            error_indices=errors
        )
    
    def evaluate_dataloader(self, model, dataloader, split):
        """Evaluate model using a DataLoader."""
        model.eval()
        model.to(self.device)
        
        all_outputs, all_targets, all_comb_ids, all_docs_function_token = [], [], [], []
        all_inputs = []
        total_sum = 0
        with torch.no_grad():
            for batch_data, batch_targets, batch_elems in dataloader:
                # Move to device
                batch_data = batch_data.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)
                batch_elems = batch_elems.to(self.device, non_blocking=True)
                full_targets = torch.cat([batch_data, batch_targets[:, -1:]], dim=1)
                
                # Get sequence info from first sample
                if not hasattr(self, '_seq_info'):
                    self._seq_info = self.token_mgr.get_seq_info(
                        batch_elems[0].cpu().numpy(),
                        self.cfg.function_type
                    )
                
                all_inputs.append(batch_data)
                # Generate predictions
                inputs = batch_data[:, :self._seq_info["prompt_pos_end"]]
                outputs = self._predict(model, inputs, self._seq_info["new_len"]+1)
                
                # Get combination IDs
                batch_np = batch_elems.cpu().numpy()
                comb_ids, docs_function_token = map_docs_to_combination_id(batch_np, self.token_mgr, self.token_map)
                
                
                all_outputs.append(outputs)
                all_targets.append(full_targets)
                all_comb_ids.append(comb_ids)
                all_docs_function_token.append(docs_function_token)
                
        
        # Concatenate all batches
        inputs = torch.cat(all_inputs, dim=0)
        outputs = torch.cat(all_outputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        combination_ids = np.concatenate(all_comb_ids)
        docs_function_token = np.concatenate(all_docs_function_token)
        # generate a unique combination ids map
        if len(all_comb_ids_map) > 1:
            combination_ids_map = {comb_id: doc_fn for doc_fn, comb_id in zip(docs_function_token, combination_ids)}
        
        with open("combination_ids_map.pkl", "wb") as f:
            pickle.dump(combination_ids_map, f)
        else:
            combination_ids_map = all_comb_ids_map[0]
        
        # Calculate metrics
        metrics = self._calc_metrics(outputs, targets, self._seq_info, combination_ids)
        
        self.logger.info(
            f"Eval {split}: Acc={metrics.sharp_accuracy:.4f} "
            f"OOD={metrics.ood_rate:.4f} MeanAcc={metrics.mean_accuracy:.4f}"
        )
        self._log_predictions(split, targets, outputs)
        self._log_predictions(split, targets, outputs)
        
        return metrics, combination_ids_map
    
    def evaluate(self, 
                verbose=False):
        
        # first load the latest checkpoint
        ckpts = self.get_latest_ckpt()
        all_results = {}
        for latest_iter, ckpt_file in ckpts:
            self.load_net(latest_iter, ckpt_file)

            # then load the dataloaders
            dataloaders = self.load_dataloaders()
            train_dataset = dataloaders["train"].dataset.data
            train_dataloader = self.create_uniform_sampler(train_dataset)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.cfg.data.batch_size,
                sampler=train_dataloader,
                num_workers=self.cfg.data.num_workers
            )
            
            results = {}
            
            for split, dataloader in dataloaders.items():
                self.logger.info(f"Evaluating {split} split...")
                metrics, combination_ids_map = self.evaluate_dataloader(self.model, dataloader, split)
                results[split] = metrics

                if verbose:
                    
                    self._log_detailed(split, metrics, latest_iter, combination_ids_map)
            
            all_results[latest_iter] = results
        return all_results
    
    def _log_detailed(self, split, metrics: Metrics, ck, combination_ids_map):
        """Log detailed evaluation results."""
        self.logger.info("=" * 60)
        self.logger.info(f"Detailed Results for {split}")
        self.logger.info("=" * 60)
        
        # Combination-wise accuracy
        if metrics.combination_sharp_acc:
            self.logger.info("\nCombination Accuracies:")
            for cid, acc in metrics.combination_sharp_acc.items():
                self.logger.info(f"  Combination {combination_ids_map[cid]} {cid}: {acc:.4f}")
        
        # Combination-wise mean accuracy
        if metrics.combination_mean_acc:
            self.logger.info("\nCombination Mean Accuracies:")
            for cid, acc in metrics.combination_mean_acc.items():
                self.logger.info(f"  Combination {combination_ids_map[cid]} {cid}: {acc:.4f}")
        
        # Overall summary
        self.logger.info(
            f"\nIter {ck} | {split} | "
            f"SharpAcc: {metrics.sharp_accuracy:.4f} | "
            f"MeanAcc: {metrics.mean_accuracy:.4f} | "
            f"OOD: {metrics.ood_rate:.4f} | "
            
        )
        self.logger.info("=" * 60)
    
    def get_output_dir(self):
        output_dir = get_directory_path(self.cfg, key='eval', prefix_dir='models/eval')
        return output_dir
    
    def save_results(self, results: Dict[str, Metrics]):
        for latest_iter, results in results.items():
            output_dir = self.get_output_dir()
            if self.cfg.sample_efficiency_experiment:
                output_dir = os.path.join(output_dir, "sample_efficiency", "nsamples_{}".format(self.cfg.nsamples))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"accs_{latest_iter}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            
            self.logger.info(f"Saved results to: {output_path}")

    def _log_predictions(self, split, inputs, outputs):
        """Log input/prediction pairs for debugging with colored error highlighting"""
        # if input data is of type torch.Tensor, convert to numpy
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        # if output is of type torch.Tensor, convert to numpy
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()

        # ANSI color codes for highlighting
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        BOLD = "\033[1m"
        END = "\033[0m"

        for idx in range(len(inputs))[:10]:
            self.logger.info("=======================================================")
            self.logger.info(f"{split}")

            # Decode input and prediction
            input_decoded = self.token_mgr.decode(inputs[idx])
            pred_decoded = self.token_mgr.decode(outputs[idx])

            self.logger.info("Input: {}".format(input_decoded))
            self.logger.info("Prediction: {}".format(pred_decoded))
            self.logger.info("=======================================================\n\n")

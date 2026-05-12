"""Data storage and directory management."""
import os
import pickle
import json
import logging
from omegaconf import OmegaConf
import numpy as np
from src.utils.storage_utils import get_directory_path

class DataStorage:
    """Handles file I/O operations and directory structure."""
    
    def __init__(self, cfg, root_dir):
        self.cfg = cfg
        self.root_dir = root_dir
        self.dir_flag = cfg.split_strategy
        self._setup_directory_paths()
        
    def _setup_directory_paths(self):
        """Compute and create directory paths."""
        base_path = get_directory_path(self.cfg, key='data', prefix_dir='data')
        self.step_fdir = os.path.join(base_path, "step_by_step", self.dir_flag)
        self.direct_fdir = os.path.join(base_path, "direct", self.dir_flag)
        if self.cfg.sample_efficiency_experiment:
            self.step_fdir = os.path.join(self.step_fdir, "sample_efficiency", "nsamples_{}".format(self.cfg.nsamples))
            self.direct_fdir = os.path.join(self.direct_fdir, "sample_efficiency", "nsamples_{}".format(self.cfg.nsamples))
        
    def store_data(self, corpus, token, token_idx, functions_info):
        """Store all generated data to disk."""
        modes = ["step_by_step", "direct"]
        
        for mode in modes:
            mode_dir = self._get_mode_directory(mode)
            self._save_mode_data(mode, mode_dir, corpus, token, 
                               token_idx, functions_info)
    
    def _get_mode_directory(self, mode):
        """Get directory path for a specific mode."""
        if mode == "step_by_step":
            return self.step_fdir
        elif mode == "direct":
            return self.direct_fdir
    
    def _save_mode_data(self, mode, mode_dir, corpus, token, 
                       token_idx, functions_info):
        """Save data for a specific mode."""
        
        
        os.makedirs(mode_dir, exist_ok=True)
        
        # Save token mappings
        pickle.dump(token_idx, open(mode_dir + "/token_idx.pkl", "wb"))
        pickle.dump(token, open(mode_dir + "/token.pkl", "wb"))
        pickle.dump(functions_info, open(mode_dir + "/functions_info.pkl", "wb"))
        
        # Save corpus data
        np.save(mode_dir + "/train_{}_corpus.npy".format(mode),
                corpus["train_" + mode])
        np.save(mode_dir + "/test_{}_corpus.npy".format(mode),
                corpus["test_" + mode])
        np.save(mode_dir + "/train_heldout_{}_corpus.npy".format(mode),
                corpus["train_heldout_" + mode])
        
        # Save config
        config_dict = self._get_config_for_mode(mode)
        json.dump(config_dict, open(mode_dir + "/config.json", "w"), indent=4)
    
    def _get_config_for_mode(self, mode):
        """Get configuration dictionary for a specific mode."""
        cfg_copy = self.cfg.copy()
        cfg_copy["tag"] = mode
        
        if mode == "step_by_step":
            cfg_copy["direct"] = False
        elif mode == "direct":
            cfg_copy["direct"] = True
            
        return OmegaConf.to_container(cfg_copy, resolve=True)
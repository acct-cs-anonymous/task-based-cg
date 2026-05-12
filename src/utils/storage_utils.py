import os
from init import ROOT_DIR


def _get_train_split(cfg) -> str:
    """Get train split from config, falling back to split_strategy if not present."""
    if hasattr(cfg, 'train_split'):
        return cfg.train_split
    elif hasattr(cfg, 'split_strategy'):
        return cfg.split_strategy
    else:
        raise ValueError("No train split found in config")


def _get_base_name(cfg) -> str:
    """Generate base directory name from config parameters."""
    return f"nalph_{cfg.n_alphabets}_seqlen_{cfg.seq_len}_fnlen_{cfg.n_functions}_taskmaxlen_{cfg.task_max_length}"


def _get_model_suffix(cfg) -> str:
    """Get model-specific directory suffix (pretrained model name or architecture details)."""
    if cfg.pretrained:
        return cfg.model_name
    else:
        return os.path.join(
            cfg.pos_embedding_type,
            cfg.nheads_nlayers,
            f"seed_{cfg.seed}",
        )


def _build_base_path(cfg, prefix_dir: str) -> str:
    """Build the base directory path common to all keys."""
    function_type = cfg.function_type
    prompt_length = cfg.prompt_length
    base_name = _get_base_name(cfg)
    
    return os.path.join(
        ROOT_DIR,
        prefix_dir,
        function_type,
        prompt_length,
        base_name,
    )


def get_directory_path(cfg: dict, key: str, prefix_dir: str = "logs") -> str:
    """
    Generate directory path based on configuration and key type.
    
    Args:
        cfg: Configuration object with training/evaluation parameters
        key: Type of directory ('train', 'eval', or 'data')
        prefix_dir: Base directory name (default: "logs")
    
    Returns:
        Full directory path (created if it doesn't exist)
    """
    base_path = _build_base_path(cfg, prefix_dir)
    prompt_mode = cfg.prompt_mode
    train_split = _get_train_split(cfg)
    
    if key == 'data':
        output_dir = base_path
    elif key == 'train':
        output_dir = os.path.join(
            base_path,
            prompt_mode,
            f"train_{train_split}",
        )
        output_dir = os.path.join(output_dir, _get_model_suffix(cfg))
    elif key == 'eval':
        output_dir = os.path.join(
            base_path,
            prompt_mode,
            f"train_{train_split}",
            f"eval_{cfg.eval_split}",
        )
        output_dir = os.path.join(output_dir, _get_model_suffix(cfg))
    elif key == 'analysis':
        output_dir = os.path.join(
            base_path,
            prompt_mode,
            f"train_{train_split}",
            f"eval_{cfg.eval_split}",
        )
        output_dir = os.path.join(output_dir, _get_model_suffix(cfg))
    elif key == 'visualization':
        output_dir = os.path.join(
            base_path,
            prompt_mode,
            f"train_{train_split}",
            f"eval_{cfg.eval_split}",
        )
        output_dir = os.path.join(output_dir, _get_model_suffix(cfg))
    else:
        raise ValueError(f"Invalid key: {key}. Must be one of: 'train', 'eval', 'data', 'analysis', 'visualization'")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

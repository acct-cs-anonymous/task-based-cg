"""Centralized logging utilities for data generation, training, and evaluation."""
import logging
import os
from typing import Optional
from src.utils.storage_utils import get_directory_path

def setup_logger(
    log_dir: str,
    log_filename: str = "data.log",
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
    filemode: str = "w",
    propagate: bool = False,
    add_stream_handler: bool = False,
) -> logging.Logger:
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    log_file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file_path, mode=filemode)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Optionally add stream handler
    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    # Prevent propagation to root logger (optional, but recommended)
    logger.propagate = propagate
    
    return logger

def setup_data_logging(
    cfg: dict,
    log_filename: str = "data.log",
) -> logging.Logger:
    """
    Set up logging for data generation tasks.
    
    Creates a logger with the standard data generation directory structure:
    {root_dir}/logs/{function_type}/{base_name}/{tag}/{prompt_length}/model_{train_split}/{pos_embedding_type}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='data', prefix_dir='logs')
    return setup_logger(log_path, log_filename=log_filename)


def setup_training_logging(
        cfg: dict,
        log_filename: str = "train.log",
) -> logging.Logger:
    """
    Set up logging for training tasks.
    
    Creates a logger with the standard training directory structure:
    {ROOT_DIR}/logs/{function_type}/{base_name}/{tag}/{prompt_length}/model_{train_split}/{pos_embedding_type}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='train', prefix_dir='logs')
    if cfg.sample_efficiency_experiment:
        log_path = os.path.join(log_path, "sample_efficiency", "nsamples_{}".format(cfg.nsamples))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    return setup_logger(log_path, log_filename=log_filename)


def setup_evaluation_logging(
    cfg: dict,
    log_filename: str = "eval.log",
) -> logging.Logger:
    """
    Set up logging for evaluation tasks.
    
    Creates a logger with the standard evaluation directory structure:
    {log_path}/{base_name}/{tag}/{prompt_length}/model_{model_split}/eval_{eval_split}/{pos_embedding_type}/seed_{seed}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='eval', prefix_dir='logs')
    
    logger = setup_logger(log_path, log_filename=log_filename)
    logger.info("Initializing SyntheticEval...")
    logger.info(os.path.join(log_path, log_filename))
    
    return logger

def setup_analysis_logging(
    cfg: dict,
    log_filename: str = "analysis.log",
) -> logging.Logger:
    """
    Set up logging for analysis tasks.
    """
    log_path = get_directory_path(cfg, key='analysis', prefix_dir='logs')
    return setup_logger(log_path, log_filename=log_filename)

def setup_visualization_logging(
    cfg: dict,
    log_filename: str = "visualization.log",
) -> logging.Logger:
    """
    Set up logging for visualization tasks.
    """
    log_path = get_directory_path(cfg, key='visualization', prefix_dir='logs')
    return setup_logger(log_path, log_filename=log_filename)
def setup_rep_analysis_logging(
    cfg: dict,
    log_filename: str = "rep_analysis.log",
) -> logging.Logger:
    """
    Set up logging for evaluation tasks.
    
    Creates a logger with the standard evaluation directory structure:
    {log_path}/{base_name}/{tag}/{prompt_length}/model_{model_split}/eval_{eval_split}/{pos_embedding_type}/seed_{seed}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='eval', prefix_dir='logs')
    
    logger = setup_logger(log_path, log_filename=log_filename)
    logger.info("Initializing Representation Analysis...")
    logger.info(os.path.join(log_path, log_filename))
    
    return logger


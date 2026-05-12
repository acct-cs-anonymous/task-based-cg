"""
Evaluation script
"""
import argparse
from init import read_config, set_seed, ROOT_DIR
from src.utils.logging_utils import setup_evaluation_logging
from src.evaluation.evaluator import Evaluator
from src.utils.storage_utils import get_directory_path
import os
def main(cfg):
    print("Running evaluation with the following configuration:")
    # print config in a readable format
    print(cfg)
    set_seed(cfg.seed)
    logger = setup_evaluation_logging(cfg)
    evaluator = Evaluator(cfg, logger)
    metrics = evaluator.evaluate(verbose=True)
    evaluator.save_results(metrics)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="combination_2",
        help="Training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="combination_2",
        help="Model evaluation split",
    )
    parser.add_argument(
        "--nheads_nlayers",
        type=str,
        default="nh6_nl3",
        default="nh6_nl3",
        help="number of heads and layers",
    )
    parser.add_argument(
        "--pos_embedding_type", type=str, default="rel_global", help="abs or rel_global"
    )
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=2, help="max task length"
    )
    
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_for_training", type=bool, default=False)
    parser.add_argument("--sample_efficiency_experiment", type=bool, default=False)
    parser.add_argument("--nsamples", type=int, default=1)
    args = parser.parse_args()

    # over-ride some of the arguments based on the run-time args
    if args.pretrained:
        cfg = read_config(f"{ROOT_DIR}/config/eval/conf_finetune.yaml")
    else:
        cfg = read_config(f"{ROOT_DIR}/config/eval/conf.yaml") 
    cfg.pretrained = args.pretrained
    cfg.prompt_mode = args.prompt_mode
    cfg.train_split = args.train_split
    cfg.eval_split = args.eval_split
    cfg.nheads_nlayers = args.nheads_nlayers
    cfg.pos_embedding_type = args.pos_embedding_type
    cfg.num_runs = args.num_runs
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    cfg.task_max_length = args.task_max_length
    cfg.data_path = get_directory_path(cfg, key='data', prefix_dir='data')
    cfg.data_path = os.path.join(cfg.data_path, cfg.prompt_mode, cfg.train_split)
    if args.sample_efficiency_experiment:
        cfg.data_path = os.path.join(cfg.data_path, "sample_efficiency", "nsamples_{}".format(args.nsamples))
    cfg.eval_for_training = args.eval_for_training
    cfg.sample_efficiency_experiment = args.sample_efficiency_experiment
    cfg.nsamples = args.nsamples
    main(cfg)


from init import read_config, set_seed, ROOT_DIR
from src.training.training import Trainer
from src.utils.logging_utils import setup_training_logging
import argparse
import os
def main(cfg, logger):
    set_seed(cfg.seed)
    trainer = Trainer(cfg, logger)
    trainer.training_loop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--train_split", type=str, default="combination_6", help="Model training split"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument(
        "--pos_embedding_type", type=str, default="abs", help="abs or rel_global"
    )
    parser.add_argument(
        "--n_heads_nlayers",
        type=str,
        default="nh6_nl3",
        help="number of heads and layers",
    )
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=6, help="task max length"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_efficiency_experiment", type=bool, default=False, help="whether to run sample efficiency experiment"
    )
    parser.add_argument(
        "--nsamples", type=int, default=10, help="number of samples"
    )
    args = parser.parse_args()
    cfg = read_config(f"{ROOT_DIR}/config/train/conf.yaml")
    cfg.prompt_mode = args.prompt_mode
    cfg.train_split = args.train_split
    cfg.epochs = args.epochs
    cfg.task_max_length = args.task_max_length
    cfg.data.n_alphabets_seq_len_fn_len_task_max_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            cfg.n_alphabets, cfg.seq_len, cfg.n_functions, cfg.task_max_length
        )
    )
    cfg.data_path = "{}/data/{}/{}/{}/{}/{}".format(
        ROOT_DIR,
        args.function_type,
        cfg.prompt_length,
        cfg.data.n_alphabets_seq_len_fn_len_task_max_length,
        cfg.prompt_mode,
        cfg.train_split,
    )
    if args.sample_efficiency_experiment:
        cfg.data_path = os.path.join(cfg.data_path, "sample_efficiency", "nsamples_{}".format(args.nsamples))
    cfg.net.pos_embedding_type = args.pos_embedding_type
    cfg.pos_embedding_type = args.pos_embedding_type
    split_nhk_nlj = args.n_heads_nlayers.split("_")
    # extract k and j from split_nhk_nlj, number of heads and layers given as nhk_nlj
    n_heads = int(split_nhk_nlj[0].split("h")[1])
    n_layers = int(split_nhk_nlj[1].split("l")[1])
    cfg.net.n_head = n_heads
    cfg.net.n_layer = n_layers
    cfg.nheads_nlayers = args.n_heads_nlayers
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    cfg.nsamples = args.nsamples
    cfg.sample_efficiency_experiment = args.sample_efficiency_experiment
    # Initialize logger
    logger = setup_training_logging(cfg)
    main(cfg, logger)

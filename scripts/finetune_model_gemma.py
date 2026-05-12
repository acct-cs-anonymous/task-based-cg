from init import read_config, set_seed, ROOT_DIR
from src.training.finetuning import FineTuner
from src.utils.logging_utils import setup_training_logging

def main():
   cfg = read_config(f"{ROOT_DIR}/config/finetune/conf.yaml")
   set_seed(cfg.seed)

   logger = setup_training_logging(cfg, log_filename="finetune.log")
   # create the fine-tuner
   finetuner = FineTuner(cfg=cfg, logger=logger)
   # start the training loop
   finetuner.training_loop()


if __name__ == "__main__":
    main()


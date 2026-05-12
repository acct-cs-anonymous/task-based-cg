import torch
import torch.nn.functional as F
import os
import shutil
from src.data.corpus_generator.token_manager import TokenManager
from src.data.loaders import get_data_loader, SyntheticDataset

from src.training.utils import configure_optimizers, move_to_device, update_cosine_warmup_lr, log_train, log_eval, save_model
from src.utils.storage_utils import get_directory_path
from src.models.nanogpt import nanoGPT
from src.data.loaders import SyntheticDataset

from init import set_seed, ROOT_DIR

import warnings
class Trainer:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        # print the config
        self.logger.info(self.cfg)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    

    def initialize_network_and_optimizer(self):
        net = nanoGPT(self.cfg.net)
        net.to(self.device)
        if self.cfg.net.compile:
            net = torch.compile(net)
        optimizer = configure_optimizers(net, self.cfg.optimizer)
        self.logger.info("number of parameters: %.2fM" % (net.get_num_params() / 1e6,))
    
        return net, optimizer

    def sanity_checks(self, loader):
        self.dictionary = TokenManager(load_path=self.cfg.data_path)
        vocab_len = self.dictionary.get_vocab_len()
        seq_len = loader.dataset.data.shape[1]
        print(f"vocab_len: {vocab_len}")
        print(f"seq_len: {seq_len}")

        # if cfg.tag == "step_by_step":
        self.cfg.net.context_size = seq_len
        self.cfg.net.vocab_size = vocab_len

        assert self.cfg.net.vocab_size >= vocab_len
        assert self.cfg.net.context_size >= seq_len
        assert self.cfg.net.n_embd % self.cfg.net.n_head == 0

        # Check if BF16 is supported
        if not torch.cuda.is_available():
            warnings.warn("WARNING: running on CPU", UserWarning)
        else:
            if not torch.cuda.is_bf16_supported():
                warnings.warn("WARNING: running without BF16", UserWarning)

            if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                raise NotImplementedError("Flash Attention requires PyTorch >= 2.0")

    def load_dataloaders(self):
        loaders = []
        for split in ["train", "test"]:
            dataset = SyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.prompt_mode)
            loader = get_data_loader(dataset, self.cfg.data.batch_size, self.cfg.data.num_workers)
            loaders.append(loader)
        return loaders

        
    def training_loop(self):
        # load the dataloaders
        loaders = self.load_dataloaders()
        train_loader = loaders[0]

        # sanity checks
        self.sanity_checks(train_loader)

        # load the network and optimizer
        self.model, self.optimizer = self.initialize_network_and_optimizer()

        sep_pos = self.dictionary.get_sep_pos(train_loader.dataset.data[0])
        
        self.train(train_loader, loaders, sep_pos)

    def get_output_dir(self):
        output_dir = get_directory_path(self.cfg, key='train', prefix_dir='models/ckpts')
       
        if self.cfg.sample_efficiency_experiment:
            output_dir = os.path.join(output_dir, "sample_efficiency", "nsamples_{}".format(self.cfg.nsamples))
        os.makedirs(output_dir, exist_ok=True)

        return output_dir

    def train(self, train_loader, loaders, sep_pos):
        self.model.train()

        dt = torch.bfloat16 if self.cfg.bf16 else torch.float32
        device_info = (self.device, dt)
        fdir = self.get_output_dir()   

        
        lr, it = 0.0, 0
        if self.cfg.sample_efficiency_experiment and len(train_loader) < 200:
            self.cfg.epochs = int(200/len(train_loader)) * self.cfg.epochs
        total_steps = len(train_loader) * self.cfg.epochs
        train_loss = []
        save_model(self.cfg, self.model, self.optimizer, it, fdir)
        
        self.logger.info(f"Total training steps: {total_steps}")
        self.logger.info(f"Learning rate warmup steps: {self.cfg.optimizer.warmup_iters}")
        save_model(self.cfg, self.model, self.optimizer, it, fdir)
        for _ in range(self.cfg.epochs):
            for dat, targets, elems in train_loader:
                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(loaders, device_info, sep_pos)
                    log_eval(it, lr, eval_info, logger=self.logger)
                    save_model(self.cfg, self.model, self.optimizer, it, fdir)

                elif it % self.cfg.log.log_interval == 0:
                    train_loss = log_train(it, lr, train_loss, logger=self.logger)

                # Update LR
                it, lr = update_cosine_warmup_lr(
                    it, self.cfg.optimizer, self.optimizer, total_steps
                )

                self.optimizer.zero_grad(set_to_none=True)
                dat, targets = move_to_device(dat, targets, self.device)

                # Compute loss
                with torch.amp.autocast(device_type=self.device, dtype=dt):
                    logits = self.model(dat)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                    )

                    train_loss.append(loss.item())

                # Update model
                loss.backward()
                if self.cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optimizer.grad_clip
                    )

                self.optimizer.step()

        # Log one final time
        eval_info = self.evaluate(loaders, device_info, sep_pos)
        log_eval(it, lr, eval_info, logger=self.logger)
        save_model(self.cfg, self.model, self.optimizer, it, fdir)

    @torch.no_grad()
    def evaluate(self, evalLoaders, device_info, sep_pos):
        all_loss, all_acc, all_sharp_acc = [], [], []
        device, dt = device_info
        self.model.eval()
        for idx, split in enumerate(("train", "test")):
            loader = evalLoaders[idx]
            sequences, total_loss, total_acc, sharp_acc = 0.0, 0.0, 0.0, 0.0
            for dat, targets, elems in loader:
                dat, targets = move_to_device(dat, targets, device)
                inputs = dat[:, :sep_pos]
                bs = dat.size(0)
                with torch.amp.autocast(device_type=device, dtype=dt):
                    # get the logits B*T*V
                    logits = self.model(dat)
                    logits = logits[:, sep_pos:]
                    # get the targets
                    targets = targets[:, sep_pos:]
                    # reshape the logits and targets 
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    # compute the loss
                    loss = F.cross_entropy(logits, targets)
                    total_loss += loss.item() * bs
                    # compute the accuracy
                    acc = logits.argmax(-1) == targets
                    total_acc += acc.float().mean().item() * bs
                    sharp_acc += acc.all(dim=-1).float().mean().item() * bs
                    
                sequences += bs
            if sequences == 0:
                    all_loss.append(float("inf"))
                    all_acc.append(float("inf"))
                    all_sharp_acc.append(float("inf"))
            else:
                all_loss.append(total_loss / sequences)
                all_acc.append(total_acc / sequences)
                all_sharp_acc.append(sharp_acc / sequences)
        info = {
            "loss": {
                "train": all_loss[0],
                "test": all_loss[1],
            },
            "acc": {
                "train": all_acc[0],
                "test": all_acc[1],
            },
            "sharp_acc": {
                "train": all_sharp_acc[0],
                "test": all_sharp_acc[1],
            },
        }
        self.model.train()
        return info
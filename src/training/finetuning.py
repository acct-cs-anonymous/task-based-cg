import torch
import torch.nn.functional as F
import os
from src.data.corpus_generator.token_manager import TokenManager
from src.data.loaders import get_data_loader, MappedSyntheticDataset
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b, load_gemma_1b
from src.training.utils import configure_optimizers, move_to_device, update_cosine_warmup_lr, log_train, log_eval, save_model_pretrained
from src.utils.storage_utils import get_directory_path

class FineTuner:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.token_mgr = TokenManager(load_path=self.cfg.data_path)

    def load_model_and_tokenizer(self):
        if self.cfg.model_name == "llama3":
            self.model, self.tokenizer, self.config = load_llama3_8b()
        elif self.cfg.model_name == "gpt2":
            self.model, self.tokenizer, self.config = load_gpt_oss_20b()
        elif self.cfg.model_name == "granite":
            self.model, self.tokenizer, self.config = load_granite_2b()
        elif self.cfg.model_name == "gemma1":
            self.model, self.tokenizer, self.config = load_gemma_1b()
        else:
            raise ValueError(f"Model {self.cfg.model_name} not supported")
        
    def load_dictionary_and_resize_tokenizer(self):
        self.dictionary = TokenManager(load_path=self.cfg.data_path)
        additional_tokens = []
        self.token_map = {}
        # check if token has mapping in model tokenizer
        for token, idx in self.dictionary.token_idx.items():
            model_token_to_idx = self.tokenizer.convert_tokens_to_ids(token)
            if model_token_to_idx is not None and token not in ["map1", "map2", "map3", "map4", "map5", "map6"] and model_token_to_idx != 3:
                self.token_map[idx] = model_token_to_idx
            else:
                additional_tokens.append(token)
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        for token, idx in self.dictionary.token_idx.items():
            if token not in self.token_map:
                self.token_map[idx] = self.tokenizer.convert_tokens_to_ids(token)
        print(f"token_map: {self.token_map}")

    def get_latest_ckpt(self):
        def itr(file):
            return int((file.split("_")[-1]).split(".")[0])
        ckpt_dir = get_directory_path(self.cfg, key='train', prefix_dir='models/ckpts')
        # replace eval with ck
        all_files = os.listdir(ckpt_dir)
        all_files = [os.path.join(ckpt_dir, file) for file in all_files if file.endswith(".pt")]
        all_ckpt_files = [(itr(file), file) for file in all_files]
        all_ckpt_files = sorted(all_ckpt_files)
       
        return all_ckpt_files[-1]

    def load_net(self, fname):
        ckpt = torch.load(fname, weights_only=False, map_location=self.device)
        self.net_cfg = ckpt["config"]
        # load token map if it is present
        if "token_map" in ckpt:
            self.token_map = ckpt["token_map"]
        else:
            self.token_map = None
        
        # get model class based on the model name and pretrained flag
        
        self.load_model_and_tokenizer()
        self.logger.info(f"Token manager Original Tokens: {self.token_mgr.token}")
        self.logger.info(f"Token manager Original Token indices: {self.token_mgr.token_idx}")
        self.token_mgr.map_tokens(self.token_map, self.tokenizer)
        self.logger.info(f"Token map: {self.token_map}")
        self.logger.info(f"Token manager Tokens: {self.token_mgr.token}")
        self.logger.info(f"Token manager Token indices: {self.token_mgr.token_idx}")
        # resize token embeddings to match the token manager
        self.model.load_state_dict(ckpt["net"])
        self.dictionary = TokenManager(load_path=self.cfg.data_path)

    
    def load_dataloaders(self):
        loaders = []
        for split in ["train", "train_heldout", "test"]:
            dataset = MappedSyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.prompt_mode, token_map=self.token_map)
            if split == "train":
                batch_size = self.cfg.batch_size
            else:
                batch_size = self.cfg.batch_size * 100
            loader = get_data_loader(dataset, batch_size, self.cfg.num_workers)
            loaders.append(loader)
        return loaders

    def get_output_dir(self):
        output_dir = get_directory_path(self.cfg, key='train', prefix_dir='models/ckpts')
        print(f"output_dir: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    def training_loop(self):
        # first load the model and tokenizer
        if self.cfg.finetune_from_ckpt:
            it, latest_ckpt = self.get_latest_ckpt()
            self.logger.info(f"Loading checkpoint from {latest_ckpt}")
            self.load_net(latest_ckpt)
            lr_ckpt, iter_ckpt = 0.0, it
        else:
            self.load_model_and_tokenizer()
            self.logger.info(f"Loading model and tokenizer from scratch")
            # then load the dictionary and resize the tokenizer
            self.load_dictionary_and_resize_tokenizer()
            lr_ckpt, iter_ckpt = 0.0, 0

        # then load the dataloaders after the tokenizer is resized
        loaders = self.load_dataloaders()
        train_loader = loaders[0]
        if isinstance(train_loader.dataset.data[0], torch.Tensor):
            sample = train_loader.dataset.data[0].cpu().numpy()
        else:
            sample = train_loader.dataset.data[0]
        self.seq_info = self.dictionary.get_seq_info(sample, self.cfg.function_type)
        # then configure the optimizers
        self.optimizer = configure_optimizers(self.model, self.cfg.optimizer)
        self.train(train_loader, loaders, lr_ckpt, iter_ckpt)

    def train(self, train_loader, loaders, lr_ckpt, iter_ckpt):
        # set the model to training mode
        self.model.train()
        # set the device and data type
        dt = torch.bfloat16 if self.cfg.bf16 else torch.float32
        device_info = (self.cfg.device, dt)
        # initialize the learning rate and iteration
        lr, it = lr_ckpt, iter_ckpt
        # calculate the total number of steps
        total_steps = len(train_loader) * self.cfg.epochs
        train_loss = []
        # start the training loop
        save_model_pretrained(self.cfg, self.model, self.optimizer, it, self.get_output_dir(), self.token_map, self.tokenizer)
        for epoch in range(self.cfg.epochs):
            for dat, targets, elems in train_loader:
                
                # update the learning rate
                it, lr = update_cosine_warmup_lr(it, self.cfg.optimizer, self.optimizer, total_steps)
                # zero the gradients
                self.optimizer.zero_grad(set_to_none=True)
                # move the data to the device
                dat, targets = move_to_device(dat, targets, self.cfg.device)
                # compute the loss
                with torch.amp.autocast(device_type=self.cfg.device, dtype=dt):
                    logits = self.model(dat)
                    loss = F.cross_entropy(logits.logits.reshape(-1, logits.logits.size(-1)), targets.reshape(-1))
                
                train_loss.append(loss.item())
                # backward the loss
                loss.backward()
                # clip the gradients
                if self.cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optimizer.grad_clip
                    )
                # step the optimizer
                self.optimizer.step()

                # if it is time to log the training loss
                if it % self.cfg.log.log_interval == 0:
                    log_train(it, lr, train_loss, logger=self.logger)

                # if it is time to evaluate
                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(loaders[1:], device_info)
                    log_eval(it, lr, eval_info, logger=self.logger)

                if it % 100000 == 0:
                    save_model_pretrained(self.cfg, self.model, self.optimizer, it, self.get_output_dir(), self.token_map, self.tokenizer)
                

        # log the final evaluation metrics
        eval_info = self.evaluate(loaders[1:], device_info)
        log_eval(it, lr, eval_info, logger=self.logger)
        save_model_pretrained(self.cfg, self.model, self.optimizer, it, self.get_output_dir(), self.token_map, self.tokenizer)

    @torch.no_grad()
    def _predict(self, inputs, new_length):
        """Predict the next tokens."""
        self.model.eval()
        for _ in range(new_length):
            logits = self.model(inputs)
            logits = logits.logits
            next_token = torch.argmax(logits[:, -1, :], -1, keepdims=True)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs

    @torch.no_grad()
    def evaluate(self, evalLoaders, device_info):
        all_loss, all_acc, all_sharp_acc = [], [], []
        device, dt = device_info
        self.model.eval()
        for idx, split in enumerate(("train_heldout", "test")):
            loader = evalLoaders[idx]
            sequences, total_loss, total_acc, sharp_acc = 0.0, 0.0, 0.0, 0.0
            for dat, targets, elems in loader:
                dat, targets = move_to_device(dat, targets, device)
                bs = dat.size(0)
                with torch.amp.autocast(device_type=device, dtype=dt):
                    # get the logits B*T*V
                    logits = self.model(dat)
                    logits = logits.logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    total_loss += loss.item() * bs
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
                "train_heldout": all_loss[0],
                "test": all_loss[1],
            },
            "acc": {
                "train_heldout": all_acc[0],
                "test": all_acc[1],
            },
            "sharp_acc": {
                "train_heldout": all_sharp_acc[0],
                "test": all_sharp_acc[1],
            },
        }
        
        self.model.train()
        return info
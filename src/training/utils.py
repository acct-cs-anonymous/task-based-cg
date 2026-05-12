import inspect
import math
import os
import shutil
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# Optimizer
def configure_optimizers(net, optim_cfg):
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": optim_cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and torch.cuda.is_available()
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=optim_cfg.learning_rate,
        betas=(optim_cfg.beta1, optim_cfg.beta2),
        **extra_args,
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def update_cosine_warmup_lr(it, cfg, optimizer, total_steps):
    it += 1
    lr = cfg.learning_rate

    if cfg.decay_lr:
        if it < cfg.warmup_iters:
            lr = lr * (it) / cfg.warmup_iters
        else:
            num = it - cfg.warmup_iters
            decay_ratio = num / (total_steps - cfg.warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            lr = cfg.min_lr + coeff * (lr - cfg.min_lr)

    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return it, lr


# Move data
def move_to_device(dat, targets, device):

    if device == "cuda":
        dat = dat.pin_memory().cuda(non_blocking=True)
        targets = targets.pin_memory().cuda(non_blocking=True)

    return dat, targets

# Logging functions
def save_model(cfg, net, optimizer, it, fdir, token_map=None, tokenizer=None):
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": cfg,
    }
    # add token map if it is not None
    if token_map is not None:
        checkpoint["token_map"] = token_map
    if tokenizer is not None:
        checkpoint["tokenizer"] = tokenizer
    fname = os.path.join(fdir, "ckpt_" + str(it + 1) + ".pt")
    torch.save(checkpoint, fname)

def save_model_pretrained(cfg, pretrained_model, optimizer, it, fdir, token_map=None, tokenizer=None):
    # save pretrained model with extended tokenizer
    model_dir = fdir + "/pretrained_model_" + str(it + 1)
    tokenizer_dir = fdir + "/tokenizer_" + str(it + 1)
    # create directories if they don't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tokenizer_dir):
        os.makedirs(tokenizer_dir)
    pretrained_model.save_pretrained(model_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(tokenizer_dir)
    
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": cfg
    }
    if tokenizer is not None:
        checkpoint["tokenizer"] = tokenizer
    if token_map is not None:
        checkpoint["token_map"] = token_map

    torch.save(checkpoint, os.path.join(fdir, "metadata_" + str(it + 1) + ".pt"))
    print(f"Saved model to {fdir}")

def save_model_hooked_transformer(cfg, net, optimizer, it, fdir):
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": it,
        "config": cfg,
    }
    fname = os.path.join(fdir, "ckpt_" + str(it + 1) + ".pt")
    torch.save(checkpoint, fname)


# Logging functions
def save_model_transformer_program(cfg, net, optimizer, fdir):
    print("Saving model to: ", fdir)
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }

    os.makedirs(fdir, exist_ok=True)

    fname = os.path.join(fdir, "model.pt")
    torch.save(checkpoint, fname)


def log_train(it, lr, train_loss, logger=None):
    logger.info("train -- iter: %d, lr: %.6f, loss: %.4f" % (it, lr, np.mean(train_loss)))
    return list()


def log_eval(it, lr, eval_info, logger=None):
    # # Programmatically log metrics based on available keys
    # print iteration and learning rate
    logger.info("----")
    logger.info(f"Iteration: {it}, Learning rate: {lr}")
    for metric_name, metric_dict in eval_info.items():
        # Get available keys (e.g., 'train', 'train_heldout', 'test')
        available_keys = [k for k in metric_dict.keys() if metric_dict[k] is not None]
        
        if not available_keys:
            continue
        
        # Determine format precision
        precision = "3f"
        
        # Build the format string dynamically
        parts = []
        values = []
        for key in available_keys:
            parts.append(f"{key}")
            values.append(metric_dict[key])
        
        # Create the log message
        keys_str = "/".join(parts)
        values_str = "/".join([f"{v:.{precision}}" for v in values])
        
        logger.info(f"{metric_name} ({keys_str}): {values_str}")
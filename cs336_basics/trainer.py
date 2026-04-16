import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from omegaconf import DictConfig
import time
import math
import os
import contextlib
import torch.distributed as dist

from cs336_basics.lr_schedule import lr_cosine_schedule
from cs336_basics.utils import gradient_clipping, save_checkpoint
import logging
logger = logging.getLogger(__name__)

def get_infinite_batches(dataloader, epoch=0):
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    for batch in dataloader:
        yield batch


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class CausalLMTrainer:
    def __init__(self,
                 model: torch.nn.Module, 
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader, 
                 criterion,
                 optimizer,
                 global_step: int,
                 local_rank,
                 cfg: DictConfig):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.local_rank = local_rank
        self.cfg = cfg

        self.global_step = global_step
        self.current_epoch = global_step // self.cfg.training.iters_per_epoch
        self.max_step = self.cfg.training.iters_per_epoch * self.cfg.training.max_epoch
        self.saved_checkpoints = []

    def train_epoch(self):
        pass

    def train_iter_per_epoch(self, data_iterator):
        self.model.train()
        total_loss = 0
        step_trained_in_current_epoch = self.global_step % self.cfg.training.iters_per_epoch
        start_time = time.time()

        for step in range(self.cfg.training.iters_per_epoch):
            batch = next(data_iterator)
            if step < step_trained_in_current_epoch:
                continue
            current_lr = lr_cosine_schedule(self.global_step, self.cfg.training.lr, self.cfg.training.min_lr, self.cfg.training.warmup, self.max_step)
            set_learning_rate(self.optimizer, current_lr)
            input_ids = batch['input_ids'].to(self.local_rank)
            labels = batch['labels'].to(self.local_rank)

            is_accumulating = (step + 1) % self.cfg.training.gradient_accumulation_steps != 0

            if is_accumulating and hasattr(self.model, "no_sync"):
                sync_context = self.model.no_sync()
            else:
                sync_context = contextlib.nullcontext()
                
            with sync_context:
                logits = self.model(input_ids)
                step_loss = self.criterion(logits, labels)
                scaled_loss = step_loss / self.cfg.training.gradient_accumulation_steps
                scaled_loss.backward()

            if not is_accumulating:
                gradient_clipping(self.model.parameters(), self.cfg.training.max_l2_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.global_step+=1
            total_loss += step_loss.item()
            if self.local_rank==0 and self.global_step % self.cfg.logging.log_interval == 0:
                elapsed = time.time() - start_time
                time_per_step = elapsed / self.cfg.logging.log_interval
                remaining_steps = self.max_step - self.global_step
                eta_seconds = int(remaining_steps * time_per_step)
                eta_str = f"{eta_seconds // 3600}h {(eta_seconds % 3600) // 60}m {eta_seconds % 60}s"
                ppl = math.exp(step_loss.item()) if step_loss.item() < 20 else float('inf')
                
                logger.info(
                    f"Step: [{self.global_step}/{self.max_step}] | Loss: {step_loss.item():0.4f} | PPL: {ppl:0.2f} | Lr: {current_lr:.2e} | ETA: {eta_str}"
                )
                if self.cfg.logging.use_wandb:
                    wandb.log({
                        "train/step_loss": step_loss.item(),
                        "train/perplexity": ppl,
                        "train/learning_rate": current_lr,
                        "global_step": self.global_step,
                        "train/eta_seconds": eta_seconds
                    })
                start_time = time.time()
                
            if self.local_rank == 0 and self.global_step % self.cfg.training.save_interval_updates == 0:
                ckpt_path = f"{self.cfg.dataset.checkpoint_dir}/ckpt_step_{self.global_step}.pt"
                save_checkpoint(self.model, self.optimizer, self.global_step, ckpt_path)
                self.saved_checkpoints.append(ckpt_path)
                
                if len(self.saved_checkpoints) > self.cfg.training.keep_interval_updates:
                    oldest_ckpt = self.saved_checkpoints.pop(0)
                    if os.path.exists(oldest_ckpt):
                        os.remove(oldest_ckpt)
                
        return total_loss / (self.cfg.training.iters_per_epoch - step_trained_in_current_epoch)
    
    @torch.no_grad()
    def evaluation(self):
        self.model.eval()
        total_loss = 0
        
        iterator = tqdm(self.valid_loader, desc="Evaluating") if self.local_rank == 0 else self.valid_loader
        
        for batch in iterator:
            input_ids = batch['input_ids'].to(self.local_rank)
            labels = batch['labels'].to(self.local_rank)

            logits = self.model(input_ids)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
        
        local_avg_loss = total_loss / len(self.valid_loader)
        tensor_loss = torch.tensor(local_avg_loss, device=self.local_rank)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        global_avg_loss = tensor_loss.item() / dist.get_world_size()
        
        eval_ppl = math.exp(global_avg_loss) if global_avg_loss < 20 else float('inf')
        return global_avg_loss, eval_ppl


    def train(self):
        for epoch in range(self.current_epoch, self.cfg.training.max_epoch):
            self.current_epoch = epoch
            data_iterator = iter(get_infinite_batches(self.train_loader, epoch))
            self.train_iter_per_epoch(data_iterator)
            
            if self.local_rank == 0:
                step = (self.current_epoch + 1) * self.cfg.training.iters_per_epoch
                save_checkpoint(self.model, self.optimizer, step, f"{self.cfg.dataset.checkpoint_dir}/ckpt_{step}.pt")
                
            loss_eval, eval_ppl = self.evaluation()
            
            if self.local_rank == 0:
                logger.info(f"Evaluation at epoch {epoch} | loss_eval: {loss_eval:0.4f} | PPL: {eval_ppl:0.2f}")
                if self.cfg.logging.use_wandb:
                    wandb.log({
                        "validation/epoch": epoch,
                        "validation/loss": loss_eval,
                        "validation/perplexity": eval_ppl
                    })
        

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from utils.commons.hparams import set_hparams
from utils.commons.trainer import Trainer
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.distributed import init_distributed

from english_dataset_class import EnglishDiffSingerDataset
from english_dataset_config import get_config
from modules.english_frontend import EnglishFrontend

class EnglishDiffSingerTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.config = get_config(self.args.config)
        
        # Set up model
        self.build_model()
        
        # Set up data
        self.setup_data_loaders()
        
        # Set up optimizer
        self.setup_optimizer()
        
        # Load checkpoint if specified
        if self.args.resume:
            self.load_checkpoint()
    
    def build_model(self):
        """Build the DiffSinger model"""
        # Import from DiffSinger's original implementation
        from tasks.tts.diffsinger import DiffSinger
        
        self.model = DiffSinger(self.config)
        
        # Initialize model
        if self.args.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.args.local_rank], find_unused_parameters=True)
        
        print(f"Model built. Parameter count: {self.model.parameters().__sizeof__()}")
    
    def setup_data_loaders(self):
        """Set up data loaders for training and validation"""
        # Training dataset
        self.train_dataset = EnglishDiffSingerDataset(
            self.config,
            self.config.processed_data_dir,
            split='train'
        )
        
        # Validation dataset
        self.val_dataset = EnglishDiffSingerDataset(
            self.config,
            self.config.processed_data_dir,
            split='val'
        )
        
        # Set up data loaders
        if self.args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        else:
            train_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.valid_batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    def setup_optimizer(self):
        """Set up optimizer and scheduler"""
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-6
        )
        
        # Set up scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.config.lr_decay
        )
    
    def load_checkpoint(self):
        """Load checkpoint if specified"""
        if os.path.exists(self.args.resume):
            load_ckpt(self.model, self.optimizer, self.args.resume)
            print(f"Loaded checkpoint from {self.args.resume}")
        else:
            print(f"No checkpoint found at {self.args.resume}")
    
    def train_step(self, batch):
        """Run one training step"""
        # Move batch to device
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(batch)
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update learning rate
        self.scheduler.step()
        
        # Log metrics
        metrics = {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        for k, v in outputs.items():
            if k != 'loss' and isinstance(v, torch.Tensor):
                metrics[k] = v.item()
        
        return metrics
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                val_losses.append(loss.item())
        
        self.model.train()
        return {'val_loss': sum(val_losses) / len(val_losses)}
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.work_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'step_{step}.pt')
        
        if isinstance(self.model, DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        torch.save({
            'step': step,
            'model': state_dict,
            'optimizer': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"Saved checkpoint at step {step} to {checkpoint_path}")
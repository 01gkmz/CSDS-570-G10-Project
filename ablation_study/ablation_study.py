#!/usr/bin/env python3
"""
Automated ablation study script for improved diffusion models.
Runs multiple experiment configurations in sequence.
"""

import argparse
import os
import json
import time
import numpy as np
import torch
import torch.distributed as dist
import copy
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import subprocess
import sys

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from improved_diffusion.resample import create_named_schedule_sampler


class CustomTrainLoop:
    """A simplified training loop that correctly updates step counts"""
    def __init__(self, model, diffusion, data, batch_size, lr, ema_rate, log_interval, save_interval, device):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.device = device
        self.step = 0  # Initialize step counter
        
        # Setup optimizer
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Setup EMA
        if isinstance(ema_rate, str):
            self.ema_rate = [float(x) for x in ema_rate.split(",")]
        else:
            self.ema_rate = [ema_rate]
        
        self.ema_params = [
            copy.deepcopy(list(model.parameters()))
            for _ in range(len(self.ema_rate))
        ]
        
        self.log_interval = log_interval
        self.save_interval = save_interval
        
    def run_step(self, batch, cond):
        """Run a single training step"""
        self.model.train()
        
        # Move batch to device
        batch = batch.to(self.device)
        cond = {k: v.to(self.device) for k, v in cond.items()}
        
        # Select random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch.shape[0],), device=self.device)
        
        # Compute loss
        losses = self.diffusion.training_losses(self.model, batch, t, model_kwargs=cond)
        loss = losses["loss"].mean()
        
        # Optimize
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        # Update EMA parameters
        for i, rate in enumerate(self.ema_rate):
            for param, ema_param in zip(self.model.parameters(), self.ema_params[i]):
                ema_param.data.mul_(rate).add_(param.data, alpha=1 - rate)
        
        # Update step counter
        self.step += 1
        
        # Return current loss value
        return loss.item()
    
    def save(self, save_dir):
        """Save model and EMA checkpoints"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save model
        torch.save(self.model.state_dict(), os.path.join(save_dir, f"model{self.step:06d}.pt"))
        
        # Save EMA models
        for i, rate in enumerate(self.ema_rate):
            ema_model = copy.deepcopy(self.model)
            for param, ema_param in zip(ema_model.parameters(), self.ema_params[i]):
                param.data.copy_(ema_param.data)
            torch.save(ema_model.state_dict(), os.path.join(save_dir, f"ema_{rate}_{self.step:06d}.pt"))


def run_experiment(experiment_config, base_args):
    """Run a single experiment with the given configuration"""
    # Create a copy of base args and apply experiment configuration
    args = argparse.Namespace(**vars(base_args))
    for key, value in experiment_config["changes"].items():
        setattr(args, key, value)
    
    # Set up experiment name and group
    args.experiment_name = experiment_config["name"]
    
    # Set up directories
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    exp_dir = results_dir / f"{args.experiment_group}_{args.experiment_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_LOGDIR"] = str(exp_dir)
    
    # Setup device and logger
    dist_util.setup_dist()
    device = dist_util.dev()
    logger.configure()
    
    # Log experiment configuration
    print(f"\n{'='*50}")
    print(f"Running experiment: {args.experiment_name}")
    print(f"Configuration: {experiment_config['changes']}")
    print(f"{'='*50}\n")
    
    logger.log(f"Starting experiment: {args.experiment_name}")
    logger.log(f"Configuration: {experiment_config['changes']}")
    
    # Create model and diffusion
    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(device)
    
    # Load data
    logger.log("Loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )
    
    # Initialize custom training loop
    logger.log(f"Starting training for {args.train_steps} steps...")
    train_loop = CustomTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        device=device
    )
    
    # Track training dynamics
    losses = []
    
    # Run training loop with progress bar
    data_iter = iter(data)
    try:
        with tqdm(total=args.train_steps, desc=f"Training {args.experiment_name}") as pbar:
            while train_loop.step < args.train_steps:
                # Get next batch
                try:
                    batch, cond = next(data_iter)
                except StopIteration:
                    # Restart iterator if we run out of data
                    data_iter = iter(data)
                    batch, cond = next(data_iter)
                
                # Run step and get loss
                current_step = train_loop.step + 1  # Step not yet incremented
                loss = train_loop.run_step(batch, cond)
                losses.append(loss)
                
                # Log progress
                pbar.update(1)
                pbar.set_description(f"{args.experiment_name} - Step {current_step}/{args.train_steps}, Loss: {loss:.4f}")
                
                # Log to logger at intervals
                if current_step % args.log_interval == 0:
                    logger.logkv("step", current_step)
                    logger.logkv("loss", loss)
                    logger.dumpkvs()
                
                # Save checkpoint at intervals
                if current_step % args.save_interval == 0 or current_step == args.train_steps:
                    logger.log(f"Saving model at step {current_step}")
                    train_loop.save(exp_dir)
                    
                    # Save loss curve
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(losses) + 1), losses, marker='.', linestyle='-', alpha=0.7)
                    plt.title(f"Training Loss - {args.experiment_name}")
                    plt.xlabel("Training Step")
                    plt.ylabel("Loss")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(exp_dir / "loss_curve.png")
                    plt.close()
                
                # Generate samples at sample_interval
                if current_step % args.sample_interval == 0 or current_step == args.train_steps:
                    logger.log(f"Generating samples at step {current_step}")
                    
                    # Generate samples
                    with torch.no_grad():
                        model.eval()
                        
                        # Generate a batch of samples
                        sample = diffusion.p_sample_loop(
                            model,
                            (16, 3, args.image_size, args.image_size),
                            clip_denoised=args.clip_denoised,
                            progress=True
                        )
                        
                        # Save a grid of samples
                        sample_dir = exp_dir / "samples"
                        sample_dir.mkdir(exist_ok=True)
                        
                        # Convert to numpy and rescale
                        samples = []
                        for i in range(sample.shape[0]):
                            img = ((sample[i] + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                            samples.append(img)
                        
                        # Create grid
                        grid_size = int(np.ceil(np.sqrt(len(samples))))
                        fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
                        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
                        
                        for i in range(grid_size * grid_size):
                            if i < len(samples):
                                axes[i].imshow(samples[i])
                            axes[i].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(sample_dir / f"sample_grid_step_{current_step:06d}.png")
                        plt.close()
                        
                        model.train()
    
    except KeyboardInterrupt:
        logger.log("Training interrupted by user")
        # If interrupted, still try to save results
    
    # Save final loss curve
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(losses) + 1), losses, marker='.', linestyle='-', alpha=0.7)
        plt.title(f"Training Loss - {args.experiment_name}")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(exp_dir / "final_loss_curve.png")
        plt.close()
    
    # Save summary results
    summary = {
        "experiment_name": args.experiment_name,
        "experiment_group": args.experiment_group,
        "train_steps_completed": train_loop.step,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses and losses[0] != 0 else None,
        "configuration": experiment_config["changes"],
        "model_params": sum(p.numel() for p in model.parameters()),
    }
    
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    logger.log(f"Experiment {args.experiment_name} completed!")
    
    # Return results
    return {
        "experiment_name": args.experiment_name,
        "final_loss": losses[-1] if losses else None,
        "loss_reduction": (losses[0] - losses[-1]) / losses[0] if losses and losses[0] != 0 else None,
    }


def run_all_experiments():
    """Run all configured experiments in sequence"""
    # Parse base arguments
    parser = create_argparser()
    args = parser.parse_args()
    
    # Define experiment configurations
    experiment_configs = [
        # Channels experiments
        {"name": "channels_32", "changes": {"num_channels": 32, "num_res_blocks": 2, "attention_resolutions": "16"}},
        {"name": "channels_64", "changes": {"num_channels": 64, "num_res_blocks": 2, "attention_resolutions": "16"}},
        
        # Residual blocks experiments
        {"name": "res_blocks_1", "changes": {"num_channels": 32, "num_res_blocks": 1, "attention_resolutions": "16"}},
        {"name": "res_blocks_2", "changes": {"num_channels": 32, "num_res_blocks": 2, "attention_resolutions": "16"}},
        
        # Attention mechanism experiments
        {"name": "attention", "changes": {"num_channels": 32, "num_res_blocks": 2, "attention_resolutions": "16"}},
        {"name": "no_attention", "changes": {"num_channels": 32, "num_res_blocks": 2, "attention_resolutions": ""}},
    ]
    
    # Record start time
    start_time = time.time()
    
    # Run each experiment
    results = []
    for config in experiment_configs:
        try:
            result = run_experiment(config, args)
            results.append(result)
        except Exception as e:
            print(f"Error in experiment {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Record end time
    total_time = time.time() - start_time
    
    # Create summary report
    summary_dir = Path(args.results_dir) / "all_experiments_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary table
    with open(summary_dir / "experiments_summary.md", "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"Total time: {total_time/3600:.2f} hours\n\n")
        f.write("| Experiment | Final Loss | Loss Reduction (%) |\n")
        f.write("|------------|------------|--------------------|\n")
        for result in results:
            f.write(f"| {result['experiment_name']} | {result['final_loss']:.6f} | {result['loss_reduction']*100:.2f} |\n")
    
    # Create summary plots
    experiment_names = [r["experiment_name"] for r in results]
    final_losses = [r["final_loss"] for r in results]
    loss_reductions = [r["loss_reduction"]*100 for r in results]
    
    # Plot final losses
    plt.figure(figsize=(12, 6))
    plt.bar(experiment_names, final_losses, color='skyblue')
    plt.xlabel('Experiment')
    plt.ylabel('Final Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(summary_dir / "final_loss_comparison.png")
    plt.close()
    
    # Plot loss reductions
    plt.figure(figsize=(12, 6))
    plt.bar(experiment_names, loss_reductions, color='lightgreen')
    plt.xlabel('Experiment')
    plt.ylabel('Loss Reduction (%)')
    plt.title('Loss Reduction Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(summary_dir / "loss_reduction_comparison.png")
    plt.close()
    
    print(f"\nAll experiments completed! Summary saved to {summary_dir}")
    return results


def create_argparser():
    defaults = dict(
        data_dir="./cifar_train",  # Default data directory
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,  # Save checkpoint every 1000 steps
        sample_interval=10000,  # Generate samples every 1000 steps
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip_denoised=True,
        results_dir="./ablation_results",
        experiment_group="model_structure",
        train_steps=30000,  # Train for 10000 steps
        image_size=32,  # CIFAR-10 image size
        num_channels=32,  # Default number of channels
        num_res_blocks=2,  # Default number of residual blocks
        attention_resolutions="16",  # Default attention resolution
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    try:
        results = run_all_experiments()
    except Exception as e:
        print(f"Error in ablation study: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保在出错时也能清理资源
        if dist.is_initialized():
            dist.destroy_process_group()
import os
import wandb
import torch
import json
from tqdm import tqdm
from collections import deque
from wandb.sdk.wandb_run import Run

class WandbTrainingManager:
    """
    Wrapper class for Weights & Biases functionality to simplify logging and tracking
    during model training.
    """
    
    def __init__(self, config, model, premade_run : Run | None = None, locked_params : list[str] | None = None):
        """
        Initializes WandB run with appropriate configuration and metric definitions.
        Also logs model architecture and parameter counts.
        """
        self.config = config
        
        self.total_params = sum(p.numel() for p in model.parameters())
        
        total_params_tmp, prefix = sum(p.numel() for p in model.parameters()), ""
        
        for value in ["K", "M", "B", "T"]:
            if total_params_tmp < 1000:
                break
            total_params_tmp /= 1000
            prefix = value
        
        total_params_shorthand = f"{total_params_tmp:.1f}{prefix}"
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if not self.config.get('wandb_run_name'):
            model_type = self.config.get('model_type', 'UNKNOWN')
            self.config['wandb_run_name'] = f"{model_type} {total_params_shorthand}"
            
        if not self.config.get('wandb_project'):
            self.config['wandb_project'] = "CSC7809 Project 2" + (" [DEBUG]" if self.config.get('debug_mode', False) else "")
        
        self.run_prefix = self.config['wandb_run_name']
        self.config = config
        self.run_dir = os.path.join(self.config['checkpoint_path'], self.config['wandb_run_name'])
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Define custom configuration for wandb charts and metrics
        wandb_config = self.config.copy()
        
        # Add additional configurations for wandb visualization
        wandb_config.update({
            "log_watch_gradients": True,
            "track_parameter_distributions": True,
            "track_gradient_distributions": True,
            "track_activation_histograms": self.config.get("track_activation_histograms", False),  # Optional and more expensive
            "track_gpu_utilization": True,
            "parameter_histogram_freq": self.config.get("parameter_histogram_freq", 100)  # Every 100 steps
        })
        
        if premade_run is None:
            self.run = wandb.init(
                project=self.config['wandb_project'],
                entity=self.config.get('wandb_entity'), # Use .get for optional keys
                config=wandb_config,
                name=self.config['wandb_run_name'],
                dir=self.config['checkpoint_path'],
                # Add tags for better organization
                tags=[
                    f"layers_{self.config['num_layers']}",
                    f"d_model_{self.config['hidden_dim']}",
                    f"batch_{self.config['batch_size']}",
                    f"seq_len_{self.config['seq_length']}",
                    "token_prediction"
                ]
            )
        else:
            self.run = premade_run
            
            # Add base tags if not already present
            base_tags = [
                f"layers_{self.config['num_layers']}",
                f"d_model_{self.config['hidden_dim']}",
                f"batch_{self.config['batch_size']}",
                f"seq_len_{self.config['seq_length']}",
                "token_prediction" # Base tag for pretraining
            ]
            current_tags = list(self.run.tags or ())
            for tag in base_tags:
                if tag not in current_tags:
                    current_tags.append(tag)
            self.run.tags = tuple(current_tags)
            
            self.run.name = self.config['wandb_run_name']
            locked_params = [] if locked_params is None else locked_params
            locked_params += ["wandb_run_name", "wandb_project", "wandb_entity", "checkpoint_path"]
            self.run.config.update({k: v for k, v in wandb_config.items() if k not in locked_params})
            
        
        # Configure wandb to create custom panels and visualizations
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("train/perplexity", summary="min")
        wandb.define_metric("train/accuracy", summary="max")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("val/perplexity", summary="min") 
        wandb.define_metric("val/accuracy", summary="max")
        wandb.define_metric("test/perplexity", summary="min")
        wandb.define_metric("test/accuracy", summary="max")
        
        
        
        # Define step metrics (ensure global_step increments across stages)
        wandb.define_metric("*", step_metric="global_step")
        
        print(f"WandB run initialized: {wandb.run.name} (ID: {wandb.run.id})")
        print(f"Run directory: {self.run_dir}")
        
        print(f"\nModel Architecture:\n{model}")
        print(f"\nTotal parameters: {self.total_params}")
        print(f"Trainable parameters: {total_trainable_params}\n")
        
        # Add model architecture summary to wandb
        wandb.summary['model_architecture'] = str(model)
        wandb.summary['total_params'] = self.total_params
        wandb.summary['trainable_params'] = total_trainable_params
        
        
        # Add parameter count by layer type
        layer_type_counts = {}
        for name, module in model.named_modules():
            layer_type = module.__class__.__name__
            if layer_type not in layer_type_counts:
                layer_type_counts[layer_type] = 0
            layer_type_counts[layer_type] += sum(p.numel() for p in module.parameters(recurse=False))
        
        wandb.summary['layer_type_counts'] = layer_type_counts
        
        # Enable model watching to track gradients and parameters
        log_freq = self.config.get('log_interval', 10) * self.config.get('gradient_accumulation_steps', 1)
        wandb.watch(
            model, 
            log="all", 
            log_freq=log_freq,
            log_graph=True
        )
        
        # Log initial parameter distributions as histograms
        if self.config.get('track_parameter_distributions', False):
            param_histograms = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Flatten and convert to numpy for histogram
                    values = param.data.flatten().cpu().numpy()
                    param_histograms[f"parameters/{name}"] = wandb.Histogram(values)
            
            wandb.log(param_histograms)
        
        self.total_trainable_params = total_trainable_params
        
        self.summary_metrics_cache = {}
        self.global_step = 0 # Initialize global step counter
        
    
    def log_dataset_signatures(self, signatures):
        """Logs dataset signatures for reproducibility."""
        wandb.config.update({"dataset_signatures": signatures})
        sig_path = os.path.join(self.run_dir, "dataset_signatures.json")
        with open(sig_path, "w") as f:
            json.dump(signatures, f, indent=4)
    
    def log_checkpoint(self, checkpoint_path, epoch, step, is_best=False, is_final=False, task_type=None):
        """Logs checkpoint metadata to wandb summary without uploading the file."""
        log_prefix = f"finetune_{task_type}_" if task_type else "pretrain_"
        
        if is_best:
            wandb.run.summary.update({
                f"{log_prefix}best_checkpoint": checkpoint_path,
                f"{log_prefix}best_checkpoint_epoch": epoch,
                f"{log_prefix}best_checkpoint_step": step
            })
        elif is_final:
            wandb.run.summary.update({
                f"{log_prefix}final_checkpoint": checkpoint_path,
                f"{log_prefix}final_checkpoint_epoch": epoch,
                f"{log_prefix}final_checkpoint_step": step
            })
        else: # Log latest checkpoint info
            wandb.run.summary.update({
                f"{log_prefix}latest_checkpoint": checkpoint_path,
                f"{log_prefix}latest_checkpoint_epoch": epoch,
                f"{log_prefix}latest_checkpoint_step": step
            })
    
    def log_train_batch_metrics(self, metrics, global_step, epoch_progress):
        """Logs training metrics for a batch."""
        log_metrics = {
            "train/loss": metrics["loss"],
            "train/perplexity": metrics["perplexity"],
            "train/accuracy": metrics["accuracy"],
        }
        
        log_data = {
            **log_metrics,
            "train/batch_tokens": metrics.get("total_tokens", 0),
            "train/learning_rate": metrics.get("learning_rate", 0),
            "epoch": epoch_progress,
            "global_step": global_step
        }
        self.update_summary(log_metrics)
        
        # Add resource metrics if available
        if "gpu_memory_allocated" in metrics:
            log_data.update({
                "system/gpu_memory_allocated_gb": metrics["gpu_memory_allocated"],
                "system/gpu_memory_reserved_gb": metrics.get("gpu_memory_reserved", 0),
                "system/gpu_max_memory_gb": metrics.get("gpu_max_memory", 0),
            })
        
        # Add performance metrics if available
        if "tokens_per_second" in metrics:
            log_data.update({
                "performance/tokens_per_second": metrics["tokens_per_second"],
                "performance/avg_sequence_length": metrics.get("avg_sequence_length", 0),
                "performance/forward_time_ms": metrics.get("forward_time_ms", 0),
                "performance/backward_time_ms": metrics.get("backward_time_ms", 0),
                "performance/optimizer_time_ms": metrics.get("optimizer_time_ms", 0),
                "performance/batch_time_ms": metrics.get("batch_time_ms", 0),
            })
            self.update_summary(log_metrics)
        
        # Add gradient metrics if available
        if "grad_norm" in metrics:
            log_data["gradients/norm"] = metrics["grad_norm"]
            self.update_summary(log_metrics)
        wandb.log(log_data)
    
    def log_validation_metrics(self, metrics, global_step, epoch_progress):
        """Logs validation metrics (generic for pretraining or finetuning)."""
        # Prefix keys with 'val/' automatically
        log_metrics = {f"val/{k}": v for k, v in metrics.items()}
        wandb.log({
            **log_metrics,
            "epoch": epoch_progress,
            "global_step": global_step
        })
        self.update_summary(log_metrics)
    
    def log_test_metrics(self, metrics, epoch, global_step, epoch_time=None):
        """Logs test metrics (generic for pretraining or finetuning)."""
        # Prefix keys with 'test/' automatically
        log_metrics = {f"test/{k}": v for k, v in metrics.items()}
        log_data = {
            **log_metrics,
            "test/total_tokens": metrics.get("total_tokens", 0), # Keep if available
            "epoch": epoch,
            "global_step": global_step,
        }
        self.update_summary(log_metrics)
        if epoch_time is not None:
            log_data.update({
                "time/epoch_seconds": epoch_time,
                "time/epoch_minutes": epoch_time / 60.0
            })
        
        wandb.log(log_data)
    
    def log_parameter_distributions(self, model, step):
        """Log parameter distributions as histograms to wandb."""
        if not self.config.get('track_parameter_distributions', False):
            return
        
        # Check if we should log based on frequency
        param_histogram_freq = self.config.get('parameter_histogram_freq', 100)
        if step % param_histogram_freq != 0:
            return
        
        param_histograms = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Flatten and convert to numpy for histogram
                values = param.data.flatten().cpu().numpy()
                param_histograms[f"parameters/{name}"] = wandb.Histogram(values)
                
                # Also track statistics
                param_histograms[f"parameter_stats/{name}_mean"] = values.mean()
                param_histograms[f"parameter_stats/{name}_std"] = values.std()
                param_histograms[f"parameter_stats/{name}_min"] = values.min()
                param_histograms[f"parameter_stats/{name}_max"] = values.max()
        
        wandb.log(param_histograms, commit=False)  # Don't commit to avoid duplicate steps
    
    def log_gradient_norms(self, grad_norms_by_layer):
        """Log gradient norms by layer."""
        if not self.config.get('track_gradient_distributions', False):
            return
        
        gradient_data = {f"gradient_norms/{name}": norm for name, norm in grad_norms_by_layer.items()}
        wandb.log(gradient_data, commit=False)  # Don't commit to avoid duplicate steps
    
    def update_best_model(self, model_path, metric_value, metric_name="perplexity"):
        """Update the best model in wandb summary."""
        wandb.run.summary[f"best_{metric_name}"] = metric_value
        wandb.run.summary["best_model_path"] = model_path
    
    def save_metrics_tracker(self, metrics_tracker):
        """Save metrics tracker to JSON and log to wandb."""
        metrics_path = os.path.join(self.run_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_tracker, f, indent=4)
    
    def save_run_information(self, run_info):
        """Save run information to JSON and log to wandb."""
        run_info_path = os.path.join(self.run_dir, "run_information.json")
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)
        
        # Log run information to wandb summary
        wandb.run.summary.update(run_info)
    
    def finish(self) -> str:
        """Finish the wandb run. Returns the run id."""
        # Get the run id
        
        try:
            run_id = self.run.id
        except Exception as e:
            print(f"Error getting run id: {e}")
            run_id = None
            
        if run_id is not None:
            for key, value in self.summary_metrics_cache.items():
                wandb.summary[key] = value

        wandb.finish()

        return run_id

    
    def update_summary(self, metrics: dict):
        """Update the summary metrics cache."""
        self.summary_metrics_cache.update(metrics)
        
        try:
            run_id = self.run.id
        except Exception as e:
            print(f"Error getting run id: {e}")
            run_id = None
        
        if run_id is not None:
            for key, value in metrics.items():
                wandb.summary[key] = value
    
    # Add a method to manually increment the global step
    def step(self):
        self.global_step += 1

    def log_metrics(self, metrics: dict, step: int = None, commit: bool = True):
        """Generic method to log any dictionary of metrics."""
        log_step = step if step is not None else self.global_step
        wandb.log(metrics, step=log_step, commit=commit)
        # Update summary cache for potential final summary update
        self.update_summary(metrics)

from pathlib import Path

import torch


class Config:
    seed = 42
    wandb_project = None
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = Path("./output")
    checkpoint_interval = 20
    train_batch_size = 16
    lr = 5e-6
    kl_weight = 0.01
    clip_eps = 0.2

    group_size = 12
    rollouts_per_step = 32
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = (
        torch.device("cuda", device_index)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

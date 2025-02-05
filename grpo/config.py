from pathlib import Path

import torch
import yaml


class Config:
    seed = 42
    wandb_project = None
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    checkpoint_path = "./output"
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

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.device = (
            torch.device("cuda", self.device_index)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    def to_dict(self):
        return {
            key: getattr(self, key)
            for key in Config.__dict__.keys()
            if not key.startswith("__")
            and hasattr(self, key)
            and not callable(getattr(self, key))
        }

    def to_yaml(self, path: Path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

import json
import random
from itertools import islice
from pathlib import Path
from typing import Iterator

import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizer

from config import Config
from loss import GRPOLoss
from replay_buffer import ReplayBuffer
from train import train_batch


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


def main():
    config = Config()
    seed_all(config.seed)
    device = config.device

    reference_model, _ = load_model(config.model_name, device_map=device)
    model, tokenizer = load_model(config.model_name, device_map=device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    prompts = list(
        islice(
            filter(
                lambda x: len(x["question"]) < 128, read_jsonl("data/math_tasks.jsonl")
            ),
            64 * 1024,
        )
    )

    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=config.rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=config.clip_eps, kl_weight=config.kl_weight)

    if config.wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=config.wandb_project)

    for batch_idx, batch in enumerate(prompt_loader):
        train_batch(
            batch_idx,
            batch,
            replay_buffer,
            model,
            reference_model,
            tokenizer,
            device,
            optimizer,
            objective,
            config.group_size,
            config.max_length,
            config.temperature,
            config.top_p,
            config.train_batch_size,
            config.epochs_per_step,
            config.max_norm,
        )

        if (
            config.checkpoint_path is not None
            and config.checkpoint_interval is not None
            and (batch_idx + 1) % config.checkpoint_interval == 0
        ):
            model.save_pretrained(config.checkpoint_path / f"step_{batch_idx}")

    if config.checkpoint_path is not None:
        model.save_pretrained(config.checkpoint_path / f"step_{batch_idx}")


if __name__ == "__main__":
    main()

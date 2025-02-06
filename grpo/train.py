import numpy as np
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from grpo.replay_buffer import join_experience_batch
from grpo.rollout import rollout, sequences_log_probs


def train_batch(
    batch_idx,
    batch,
    replay_buffer,
    model,
    reference_model,
    tokenizer,
    device,
    optimizer,
    objective,
    config,
):
    rollout_returns = []
    replay_buffer.clear()

    questions = batch["question"]
    answers = batch["answer"]
    for q, a in zip(questions, answers):
        experience = rollout(
            model,
            reference_model,
            tokenizer,
            q,
            a,
            num_rollouts=config.group_size,
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
        )

        rollout_returns.append(experience.returns.cpu())
        replay_buffer.append(experience.cpu())

    torch.cuda.empty_cache()
    rollout_returns = np.array(rollout_returns).flatten()
    episode_mean = rollout_returns.mean()
    epsiode_acc = (rollout_returns > 0).mean()

    print(f"Step {batch_idx} Returns {episode_mean:.4f} Accuracy {epsiode_acc:.4f}")
    wandb.log({"return_mean": episode_mean, "return_accuracy": epsiode_acc})

    experience_sampler = DataLoader(
        replay_buffer,
        batch_size=config.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=join_experience_batch,
    )

    for step_epoch in range(config.epochs_per_step):
        model.train()

        for exp in experience_sampler:
            exp = exp.to(device)

            optimizer.zero_grad()

            log_probs = sequences_log_probs(
                model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
            )

            loss, kl = objective(log_probs=log_probs, experience=exp)

            if not loss.isfinite():
                print(
                    f"Loss not finite, skipping backward, loss={loss}"
                    f"experience.advantages={experience.advantages}"
                )
                continue

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            wandb.log({"loss": loss, "kl": kl, "grad_norm": grad_norm})
            optimizer.step()

            print(
                f"Step {step_epoch}: loss {loss:.4f} "
                f"kl {kl:.4f}, grad_norm {grad_norm:.4f}"
            )

import re

import torch
import torch.nn.functional as F
import wandb
from loss import approx_kl_divergence
from replay_buffer import Experience, join_experience_batch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import GenerationConfig, LlamaForCausalLM, PreTrainedTokenizer

# DeepSeek Zero system prompt
SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a \
question, and the Assistant solves it. The assistant first thinks about the \
reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and \
<answer> </answer> tags, respectively, i.e., <think> reasoning process here \
</think><answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    reference_model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(
        num_rollouts, 1, dtype=torch.float, device=sequence_ids.device
    )
    for i, completion in enumerate(completions):
        # search answer tag
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )

        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        returns[i] = reward

    advantages = group_advantages(returns)
    attention_mask = sequence_ids != pad_token_id

    log_probs = sequences_log_probs(
        model=model,
        sequence_ids=sequence_ids,
        attention_mask=attention_mask,
    )
    log_probs_ref = sequences_log_probs(
        model=reference_model,
        sequence_ids=sequence_ids,
        attention_mask=attention_mask,
    )
    kl = approx_kl_divergence(
        log_probs=log_probs,
        log_probs_ref=log_probs_ref,
        action_mask=action_mask,
    )

    return Experience(
        sequences=sequence_ids,
        action_log_probs=log_probs,
        log_probs_ref=log_probs_ref,
        returns=returns,
        advantages=advantages,
        attention_mask=attention_mask,
        action_mask=action_mask,
        kl=kl,
    )


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


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
    group_size,
    max_length,
    temperature,
    top_p,
    train_batch_size,
    epochs_per_step,
    max_norm,
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
            num_rollouts=group_size,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
        )

        rollout_returns.append(experience.returns.cpu())
        replay_buffer.append(experience.cpu())

    torch.cuda.empty_cache()
    episode_return_sum = torch.stack(rollout_returns).sum()

    print(f"returns of step {batch_idx}: {episode_return_sum:.4f}")
    wandb.log({"returns": episode_return_sum})

    experience_sampler = DataLoader(
        replay_buffer,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=join_experience_batch,
    )

    for step_epoch in range(epochs_per_step):
        model.train()

        for exp in experience_sampler:
            exp = exp.to(device)

            optimizer.zero_grad()

            log_probs = sequences_log_probs(
                model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
            )

            loss, kl = objective(log_probs=log_probs, experience=exp)

            if not loss.isfinite():
                print(f"Loss not finite, skipping backward, loss={loss}")
                print(f"experience.advantages={experience.advantages}")
                continue

            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
            wandb.log({"kl": kl, "grad_norm": grad_norm})
            optimizer.step()

            print(f"{step_epoch}: kl={kl: .4f}, grad_norm={grad_norm: .4f}")

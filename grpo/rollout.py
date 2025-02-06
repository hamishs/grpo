import re

import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LlamaForCausalLM, PreTrainedTokenizer

from grpo.loss import approx_kl_divergence
from grpo.replay_buffer import Experience


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

    model_inputs = prepare_model_inputs(task, tokenizer, model.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # sample completions
    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)

    input_seq_len = input_ids.shape[1]
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_seq_len:], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_seq_len:] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # determine rewards
    returns = torch.zeros(
        num_rollouts, 1, dtype=torch.float, device=sequence_ids.device
    )
    for i, completion in enumerate(completions):
        returns[i] = evaluate_response(completion, oracle_answer)

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


def prepare_model_inputs(
    task: str,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> torch.Tensor:
    # DeepSeek Zero system prompt
    SYSTEM_PROMPT = """A conversation between User and Assistant. The user asks a \
    question, and the Assistant solves it. The assistant first thinks about the \
    reasoning process in the mind and then provides the user with the answer. \
    The reasoning process and answer are enclosed within <think> </think> and \
    <answer> </answer> tags, respectively, i.e., <think> reasoning process here \
    </think><answer> answer here </answer>
    """

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

    return tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(device)


def evaluate_response(completion: str, oracle_answer: str) -> float:
    # search answer tag
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL,
    )

    answer = answer_match.group(1) if answer_match else None
    reward = 0.0
    if answer is not None:
        if answer == oracle_answer:
            reward = 1.0
        elif oracle_answer in answer:
            reward = 0.5
        else:
            reward = 0.01
    return reward


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

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import datasets
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="reinforcement learning from human feedback (PPO)")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--sft_model', default="ckpts/Llama-3.2-1B-sft", type=str)
parser.add_argument('--reward_model', default="ckpts/Llama-3.2-1B-rm", type=str)
parser.add_argument('--hf_dataset_name', default='HuggingFaceH4/ultrafeedback_binarized', type=str)
parser.add_argument('--prompt_split', default='train_gen', type=str)
parser.add_argument('--num_prompts', default=None, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--learning_rate', default=1e-6, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--ppo_epochs', default=4, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--max_new_tokens', default=512, type=int)
parser.add_argument('--clip_range', default=0.2, type=float)
parser.add_argument('--kl_coef', default=0.05, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--gae_lambda', default=0.95, type=float)
parser.add_argument('--logging_steps', default=10, type=int)
parser.add_argument('--output_dir', default="ckpts", type=str)
parser.add_argument('--output_name', default="ppo", type=str)
args = parser.parse_args()

def get_batches(dataset, batch_size):
    dataset = dataset.shuffle()
    for start in range(0, len(dataset), batch_size):
        yield dataset[start:start + batch_size]

def masked_mean(tensor, mask):
    return (tensor * mask).sum() / mask.sum().clamp(min=1)

def whiten(advantages, mask):
    mean = masked_mean(advantages, mask)
    variance = masked_mean((advantages - mean) ** 2, mask)
    return (advantages - mean) * torch.rsqrt(variance + 1e-8) * mask

def build_response_mask(response_ids, eos_token_id):
    is_eos = (response_ids == eos_token_id).int()
    after_eos = (is_eos.cumsum(dim=1) - is_eos) > 0
    return (~after_eos).float()

def compute_scores(reward_model, tokenizer, prompts, responses):
    texts = [
        tokenizer.apply_chat_template([{"role": 'user', "content": prompt}, {"role": 'assistant', "content": response}], tokenize=False)
        for prompt, response in zip(prompts, responses)
    ]
    inputs = tokenizer(texts, add_special_tokens=False, padding=True, truncation=True, max_length=args.max_length, return_tensors='pt')
    return reward_model(**inputs.to(reward_model.device)).logits.squeeze(-1).float()

def response_logprobs_and_values(model, value_head, full_ids, attention_mask, prompt_length, response_ids):
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    outputs = model(full_ids, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=value_head is not None)
    logits = outputs.logits[:, prompt_length - 1:-1, :].float()
    logprobs = torch.gather(torch.log_softmax(logits, dim=-1), 2, response_ids.unsqueeze(-1)).squeeze(-1)
    if value_head is None:
        return logprobs, None
    values = value_head(outputs.hidden_states[-1][:, prompt_length - 1:-1, :]).squeeze(-1).float()
    return logprobs, values

def gae_advantages(token_rewards, values, response_mask, gamma, gae_lambda):
    seq_length = values.shape[1]
    advantages = torch.zeros_like(values)
    last_gae = torch.zeros_like(values[:, 0])
    for step in reversed(range(seq_length)):
        next_value = values[:, step + 1] if step + 1 < seq_length else torch.zeros_like(last_gae)
        delta = token_rewards[:, step] + gamma * next_value - values[:, step]
        last_gae = delta + gamma * gae_lambda * last_gae
        advantages[:, step] = last_gae
    advantages = advantages * response_mask
    returns = advantages + values
    return advantages, returns

def train(args):
    prompts = datasets.load_dataset(args.hf_dataset_name, split=args.prompt_split).select_columns(['prompt'])
    if args.num_prompts:
        prompts = prompts.shuffle(seed=0).select(range(min(args.num_prompts, len(prompts))))

    sft_model_path = str(ROOT / args.sft_model)
    reward_model_path = str(ROOT / args.reward_model)

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.padding_side = 'left'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.float32).to(device)
    value_head = nn.Linear(model.config.hidden_size, 1).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.bfloat16).to(device)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path, num_labels=1, dtype=torch.bfloat16).to(device)
    for frozen_model in [ref_model, reward_model]:
        frozen_model.eval()
        frozen_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(value_head.parameters()), lr=args.learning_rate)

    num_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(total=num_batches * args.num_epochs, desc='Training')
    global_step = 0

    for epoch in range(args.num_epochs):
        for batch in get_batches(prompts, args.batch_size):
            global_step += 1

            prompt_texts = [
                tokenizer.apply_chat_template([{"role": 'user', "content": prompt}], add_generation_prompt=True, tokenize=False)
                for prompt in batch['prompt']
            ]
            prompt_inputs = tokenizer(prompt_texts, add_special_tokens=False, padding=True, truncation=True,
                                      max_length=args.max_length, return_tensors='pt').to(device)
            prompt_length = prompt_inputs['input_ids'].shape[1]

            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
                generated_ids = model.generate(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                response_ids = generated_ids[:, prompt_length:]
                response_mask = build_response_mask(response_ids, tokenizer.eos_token_id)
                response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                full_ids = torch.cat([prompt_inputs['input_ids'], response_ids], dim=1)
                attention_mask = torch.cat([prompt_inputs['attention_mask'], response_mask.long()], dim=1)

                raw_scores = compute_scores(reward_model, tokenizer, batch['prompt'], response_texts)
                scores = raw_scores - raw_scores.mean()

                old_logprobs, old_values = response_logprobs_and_values(model, value_head, full_ids, attention_mask, prompt_length, response_ids)
                ref_logprobs, _ = response_logprobs_and_values(ref_model, None, full_ids, attention_mask, prompt_length, response_ids)
                old_values = old_values * response_mask

                kl = (old_logprobs - ref_logprobs) * response_mask
                token_rewards = -args.kl_coef * kl
                last_token_index = (response_mask.sum(dim=1).long() - 1).clamp(min=0)
                token_rewards[torch.arange(len(scores), device=device), last_token_index] += scores

                advantages, returns = gae_advantages(token_rewards, old_values, response_mask, args.gamma, args.gae_lambda)
                advantages = whiten(advantages, response_mask)

            for ppo_epoch in range(args.ppo_epochs):
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    logprobs, values = response_logprobs_and_values(model, value_head, full_ids, attention_mask, prompt_length, response_ids)

                ratio = torch.exp(logprobs - old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
                policy_loss = masked_mean(-torch.min(ratio * advantages, clipped_ratio * advantages), response_mask)
                value_loss = masked_mean((values - returns) ** 2, response_mask)
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.logging_steps == 0:
                mean_kl = (kl.sum(dim=1)).mean().item()
                print(f"Step [{global_step}], Loss: {loss.item():.4f}, KL: {mean_kl:.4f}, Score: {raw_scores.mean().item():.3f}")

            progress_bar.update(1)

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
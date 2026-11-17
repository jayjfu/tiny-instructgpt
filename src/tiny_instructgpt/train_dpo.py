import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import datasets


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="direct preference optimization (DPO) without TRL")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--sft_model', default="ckpts/Llama-3.2-1B-sft", type=str)
parser.add_argument('--hf_dataset_name', default='HuggingFaceH4/ultrafeedback_binarized', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--learning_rate', default=5e-7, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--logging_steps', default=1_000, type=int)
parser.add_argument('--save_steps', default=5_000, type=int)
parser.add_argument('--eval_steps', default=2_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--output_dir', default="ckpts", type=str)
parser.add_argument('--output_name', default="dpo", type=str)
args = parser.parse_args()

def compute_dpo_loss(model_chosen_logps, model_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    logits = (model_chosen_logps - model_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    losses = -F.logsigmoid(beta * logits)
    return losses.mean()

class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta, max_length, **kwargs):
        super().__init__(**kwargs)
        self._signature_columns = ['prompt', 'chosen', 'rejected']

        self.ref_model = ref_model
        self.beta = beta
        self.max_length = max_length

        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        self.ref_model.to(self.model.device)

    def tokenize(self, texts, device):
        return self.processing_class(
            texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        ).to(device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompts, chosen, rejected = inputs['prompt'], inputs['chosen'], inputs['rejected']

        chosen_inputs = self.tokenize([prompt + completion for prompt, completion in zip(prompts, chosen)], model.device)
        rejected_inputs = self.tokenize([prompt + completion for prompt, completion in zip(prompts, rejected)], model.device)
        prompt_lengths = torch.tensor(
            [len(self.processing_class(prompt, add_special_tokens=False)['input_ids']) for prompt in prompts],
            device=model.device,
        )

        model_chosen_logps = self.get_response_logps(model, chosen_inputs, prompt_lengths)
        model_rejected_logps = self.get_response_logps(model, rejected_inputs, prompt_lengths)

        with torch.no_grad():
            ref_chosen_logps = self.get_response_logps(self.ref_model, chosen_inputs, prompt_lengths)
            ref_rejected_logps = self.get_response_logps(self.ref_model, rejected_inputs, prompt_lengths)

        loss = compute_dpo_loss(
            model_chosen_logps,
            model_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self.beta,
        )

        return (loss, None) if return_outputs else loss

    @staticmethod
    def get_response_logps(model, batch_inputs, prompt_lengths):
        logits = model(**batch_inputs).logits[:, :-1, :]
        labels = batch_inputs['input_ids'][:, 1:]
        token_logps = torch.gather(F.log_softmax(logits, dim=-1), 2, labels.unsqueeze(-1)).squeeze(-1)

        positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
        response_mask = (positions >= prompt_lengths.unsqueeze(1) - 1) & batch_inputs['attention_mask'][:, 1:].bool()
        return (token_logps * response_mask).sum(dim=-1)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss, None, None)

def train(args):
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)

    sft_model_path = str(ROOT / args.sft_model)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.padding_side = 'right'

    def apply_chat_template(example):
        prompt_text = tokenizer.apply_chat_template(example['chosen'][:-1], add_generation_prompt=True, tokenize=False)
        chosen_text = tokenizer.apply_chat_template(example['chosen'], tokenize=False)
        rejected_text = tokenizer.apply_chat_template(example['rejected'], tokenize=False)
        return {
            "prompt": prompt_text,
            "chosen": chosen_text[len(prompt_text):],
            "rejected": rejected_text[len(prompt_text):],
        }

    train_dataset = raw_dataset['train_prefs'].map(
        apply_chat_template,
        remove_columns=raw_dataset['train_prefs'].column_names,
    )
    eval_dataset = raw_dataset['test_prefs'].map(
        apply_chat_template,
        remove_columns=raw_dataset['test_prefs'].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.float32)
    ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.bfloat16)

    def data_collator(features):
        return {column: [feature[column] for feature in features] for column in ['prompt', 'chosen', 'rejected']}

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=True,
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_length=args.max_length,
        beta=args.beta,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import DPOTrainer, DPOConfig


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="direct preference optimization (DPO, TRL)")
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
parser.add_argument('--output_dir', default="ckpts", type=str)
parser.add_argument('--output_name', default="dpo", type=str)
args = parser.parse_args()

def to_conversational(example):
    return {
        "prompt": example['chosen'][:-1],
        "chosen": example['chosen'][-1:],
        "rejected": example['rejected'][-1:],
    }

def train(args):
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)
    train_dataset = raw_dataset['train_prefs'].map(to_conversational, remove_columns=raw_dataset['train_prefs'].column_names)
    eval_dataset = raw_dataset['test_prefs'].map(to_conversational, remove_columns=raw_dataset['test_prefs'].column_names)

    sft_model_path = str(ROOT / args.sft_model)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    model = AutoModelForCausalLM.from_pretrained(sft_model_path, dtype=torch.float32)

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    dpo_config = DPOConfig(
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
        max_length=args.max_length,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="supervised fine-tuning (TRL)")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--template_model', default='unsloth/Llama-3.2-1B-Instruct', type=str)
parser.add_argument('--dataset_file', default="data/oasst2/oasst2_messages.jsonl", type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=2e-4, type=float)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--save_steps', default=2_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="ckpts", type=str)
parser.add_argument('--output_name', default="sft", type=str)
args = parser.parse_args()

def train(args):
    dataset = datasets.load_dataset('json', data_files={"train": str(ROOT / args.dataset_file)})

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.template_model)

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=500,
        save_steps=args.save_steps,
        bf16=True,
        max_length=args.max_length,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset['train'],
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
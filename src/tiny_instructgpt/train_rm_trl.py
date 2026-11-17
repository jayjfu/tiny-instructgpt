import argparse
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import datasets
from trl import RewardConfig, RewardTrainer


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="reward model training (TRL)")
parser.add_argument('--base_model', default='unsloth/Llama-3.2-1B', type=str)
parser.add_argument('--sft_model', default="ckpts/Llama-3.2-1B-sft", type=str)
parser.add_argument('--hf_dataset_name', default='HuggingFaceH4/ultrafeedback_binarized', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--max_length', default=512, type=int)
parser.add_argument('--save_steps', default=5_000, type=int)
parser.add_argument('--no_resume', action='store_true')
parser.add_argument('--output_dir', default="ckpts", type=str)
parser.add_argument('--output_name', default="rm", type=str)
args = parser.parse_args()

def train(args):
    # Same preference pairs as DPO: full conversations, TRL applies the chat template
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)
    dataset = raw_dataset.select_columns(['chosen', 'rejected'])

    # Initialized from the SFT model, as in InstructGPT
    sft_model_path = str(ROOT / args.sft_model)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(sft_model_path, num_labels=1, dtype=torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    reward_config = RewardConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=1_000,
        save_steps=args.save_steps,
        bf16=True,
        max_length=args.max_length,
    )
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset['train_prefs'],
        eval_dataset=dataset['test_prefs'],
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
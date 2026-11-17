import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import datasets


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="supervised fine-tuning")
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
    raw_dataset = datasets.load_dataset('json', data_files={"train": str(ROOT / args.dataset_file)})

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.template_model)

    def tokenize_function(examples):
        texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in examples['messages']]
        return tokenizer(texts, add_special_tokens=False, truncation=True, max_length=args.max_length)

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset['train'].column_names,
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=500,
        save_steps=args.save_steps,
        bf16=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
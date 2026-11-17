import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import datasets


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="reward model training")
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

class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_chosen = model(
            input_ids=inputs['input_ids_chosen'],
            attention_mask=inputs['attention_mask_chosen'],
        )
        outputs_rejected = model(
            input_ids=inputs['input_ids_rejected'],
            attention_mask=inputs['attention_mask_rejected'],
        )

        reward_chosen = outputs_chosen.logits.squeeze(-1)
        reward_rejected = outputs_rejected.logits.squeeze(-1)

        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

        return (loss, {"loss": loss, "reward_chosen": reward_chosen, "reward_rejected": reward_rejected}) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs, return_outputs=False)

        if prediction_loss_only:
            return loss, None, None
        else:
            batch_size = inputs['input_ids_chosen'].shape[0]
            dummy_labels = torch.zeros(batch_size, device=loss.device)
            return loss, None, dummy_labels

def train(args):
    raw_dataset = datasets.load_dataset(args.hf_dataset_name)
    raw_dataset = datasets.DatasetDict({split: raw_dataset[split] for split in ['train_prefs', 'test_prefs']})

    sft_model_path = str(ROOT / args.sft_model)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)

    def preprocess_function(examples):
        tokenized = {}
        for column in ['chosen', 'rejected']:
            texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in examples[column]]
            encodings = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=args.max_length)
            tokenized[f"input_ids_{column}"] = encodings['input_ids']
            tokenized[f"attention_mask_{column}"] = encodings['attention_mask']
        return tokenized

    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset['train_prefs'].column_names,
    )

    def paired_collator(features):
        max_len = max(len(feature[f"input_ids_{column}"]) for feature in features for column in ['chosen', 'rejected'])

        batch = {}
        for column in ['chosen', 'rejected']:
            padded = tokenizer.pad(
                {
                    "input_ids": [feature[f"input_ids_{column}"] for feature in features],
                    "attention_mask": [feature[f"attention_mask_{column}"] for feature in features],
                },
                padding='max_length',
                max_length=max_len,
                return_tensors='pt',
            )
            batch[f"input_ids_{column}"] = padded['input_ids']
            batch[f"attention_mask_{column}"] = padded['attention_mask']
        return batch

    model = AutoModelForSequenceClassification.from_pretrained(sft_model_path, num_labels=1, dtype=torch.float32)
    model.config.pad_token_id = tokenizer.pad_token_id

    output_dir = str(ROOT / f"{args.output_dir}/{args.base_model.split('/')[-1]}-{args.output_name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='steps',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        logging_steps=1_000,
        save_steps=args.save_steps,
        bf16=True,
        remove_unused_columns=False,
    )
    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train_prefs'],
        eval_dataset=tokenized_dataset['test_prefs'],
        data_collator=paired_collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=not args.no_resume)
    trainer.save_model(output_dir)

def main():
    train(args)

if __name__ == "__main__":
    main()
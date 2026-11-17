import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="Chat with a trained model")
parser.add_argument('--model_path', default="ckpts/Llama-3.2-1B-ppo", type=str)
parser.add_argument('--prompt', default="How can I build a daily drawing habit?", type=str)
parser.add_argument('--max_new_tokens', default=512, type=int)
parser.add_argument('--temperature', default=0.7, type=float)
args = parser.parse_args()

def main():
    model_path = str(ROOT / args.model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)

    messages = [{"role": 'user', "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
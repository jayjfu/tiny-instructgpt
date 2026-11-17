import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = "You are a helpful assistant."
ROLE_MAPPING = {"prompter": 'user', "assistant": 'assistant'}

parser = argparse.ArgumentParser(description="Convert OASST2 trees into chat-message conversations")
parser.add_argument('--input_file', default="data/oasst2/2023-11-05_oasst2_ready.trees.jsonl", type=str)
parser.add_argument('--output_file', default="data/oasst2/oasst2_messages.jsonl", type=str)
parser.add_argument('--language', default='en', type=str)
args = parser.parse_args()

def reply_rank(reply):
    return reply['rank'] if reply.get('rank') is not None else float('inf')

def best_branch(node):
    messages = []
    while node is not None:
        messages.append({"role": ROLE_MAPPING[node['role']], "content": node['text']})
        replies = node.get('replies', [])
        node = min(replies, key=reply_rank) if replies else None

    if messages[-1]['role'] == 'user':
        messages = messages[:-1]
    return messages

def main():
    output_path = ROOT / args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.input_file) as input_file, open(output_path, 'w') as output_file:
        for line in input_file:
            root = json.loads(line)['prompt']
            if args.language and root.get('lang') != args.language:
                continue
            messages = best_branch(root)
            if len(messages) < 2:
                continue
            if SYSTEM_PROMPT:
                messages = [{"role": 'system', "content": SYSTEM_PROMPT}] + messages
            output_file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
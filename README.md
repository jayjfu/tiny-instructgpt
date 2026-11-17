# Tiny InstructGPT

### A tiny reproduction of InstructGPT: post-training Llama-3.2-1B into a chat assistant with SFT, reward modeling and PPO, plus DPO as a modern alternative.

Every training stage comes in two versions: one built on [TRL](https://github.com/huggingface/trl), and one written directly on `transformers.Trainer` so the losses and training loops are easy to read.

## Pipelines

- **RLHF:** Llama-3.2-1B → SFT (OASST2) → Reward Model (UltraFeedback `train_prefs`) → PPO (UltraFeedback `train_gen` prompts)
- **DPO:** Llama-3.2-1B → SFT (OASST2) → DPO (UltraFeedback `train_prefs`)

## Project Structure
```
tiny-instructgpt
├── ckpts/                      # model checkpoints (created during training)
├── data/                       # downloaded + processed data
│   └── README.md
├── scripts
│   ├── get_data.sh             # download + prepare OASST2
│   ├── run_rlhf.sh             # SFT → RM → PPO
│   └── run_dpo.sh              # SFT → DPO
├── src
│   └── tiny_instructgpt
│       ├── prepare_oasst2.py   # OASST2 trees → chat messages
│       ├── train_sft.py        train_sft_trl.py
│       ├── train_rm.py         train_rm_trl.py
│       ├── train_ppo.py
│       ├── train_dpo.py        train_dpo_trl.py
│       └── chat.py             # inference
├── CHANGELOG.md
├── README.md
└── requirements.txt
```
Each stage has a from-scratch version (`train_*.py`) and a TRL version (`train_*_trl.py`). Both save to the same checkpoint folder, so either one can feed the next stage. All stages use the Llama-3.2-Instruct chat template.

## Quick Start
```bash
bash scripts/get_data.sh     # download + prepare OASST2
bash scripts/run_rlhf.sh     # SFT → RM → PPO
bash scripts/run_dpo.sh      # SFT → DPO

# Use the from-scratch implementations instead of TRL
USE_TRL=false bash scripts/run_rlhf.sh
```
Checkpoints are saved under `ckpts/` (e.g. `ckpts/Llama-3.2-1B-sft`, `ckpts/Llama-3.2-1B-rm`, `ckpts/Llama-3.2-1B-ppo`).

## Data Preparation
OASST2 is downloaded and converted into chat-message conversations, following the highest-ranked reply at each turn:
```bash
bash scripts/get_data.sh
```
Details of the data are in [`data/README.md`](data/README.md).

## Step by Step
```bash
cd src/tiny_instructgpt   # checkpoint paths stay relative to the project root
```

### Step 1: Supervised Fine-Tuning
```bash
python train_sft_trl.py \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --num_epochs 2 \
  --max_length 512 \
  --save_steps 2_000 \
  --no_resume
```

<details><summary> From scratch (without TRL): </summary>

```bash
python train_sft.py \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --num_epochs 2 \
  --max_length 512 \
  --save_steps 2_000 \
  --no_resume
```
</details>

### Step 2: Reward Model
```bash
python train_rm_trl.py \
  --sft_model ckpts/Llama-3.2-1B-sft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --save_steps 5_000 \
  --no_resume
```

<details><summary> From scratch (without TRL): </summary>

```bash
python train_rm.py \
  --sft_model ckpts/Llama-3.2-1B-sft \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --max_length 512 \
  --save_steps 5_000 \
  --no_resume
```
</details>

### Step 3: PPO
```bash
python train_ppo.py \
  --sft_model ckpts/Llama-3.2-1B-sft \
  --reward_model ckpts/Llama-3.2-1B-rm \
  --batch_size 2 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --ppo_epochs 4 \
  --max_new_tokens 512 \
  --clip_range 0.2 \
  --kl_coef 0.05 \
  --logging_steps 10
```

### Alternative: DPO
```bash
python train_dpo_trl.py \
  --sft_model ckpts/Llama-3.2-1B-sft \
  --batch_size 2 \
  --learning_rate 5e-7 \
  --num_epochs 1 \
  --max_length 512 \
  --logging_steps 1_000 \
  --save_steps 5_000 \
  --eval_steps 2_000 \
  --no_resume
```

<details><summary> From scratch (without TRL): </summary>

```bash
python train_dpo.py \
  --sft_model ckpts/Llama-3.2-1B-sft \
  --batch_size 2 \
  --learning_rate 5e-7 \
  --num_epochs 1 \
  --max_length 512 \
  --logging_steps 1_000 \
  --save_steps 5_000 \
  --eval_steps 2_000 \
  --no_resume
```
</details>

## Inference
```bash
python chat.py --model_path ckpts/Llama-3.2-1B-ppo --prompt "How can I build a daily drawing habit?"
```

## Sample Outputs

**User Prompt:** 
```text
How can I build a daily drawing habit?
```

**Model Output:**

```text
It's important to understand that creating a daily drawing habit takes time and effort, but it can be done by prioritizing the most important aspects of the craft and making it a priority. Here are some tips that can help you build a daily drawing habit:

1. **Set a goal:** Start by setting a specific goal for how many drawings you want to complete each day. This can be anything from a few drawings to a specific style or subject.

2. **Find a routine:** Consistency is key. Try to go to the same time and place every day, even on weekends. This will help you develop a routine that you can stick to even on weekends.

3. **Set a goal:** Give yourself a goal for the number of drawings you want to complete each day. This will help you stay motivated and track your progress.

4. **Create a schedule:** Plan out your drawings in advance. This can help you stay organized and ensure that you don't waste time trying to draw something that you forgot to complete the previous day.

5. **Keep a journal:** Writing down your drawings can help you reflect on what you've done well and what you can do even better. This can also help you identify any patterns or themes that you need to work on.

6. **Stay hydrated:** Drinking plenty of water can help you stay focused and motivated. You can also use a water bottle or a cup of water to help you stay hydrated.

7. **Take breaks:** It's important to take regular breaks to recharge your body and mind. This can help you stay focused and motivated.

8. **Use a drawing app:** There are many drawing apps available that can help you stay organized and track your progress. Some popular ones include Adobe Photoshop, Canva, and Sketch.

9. **Learn a new skill:** If you've never tried drawing, there are many online resources that can help you learn a new skill. Some popular ones include Udemy and Coursera.

10. **Practice regularly:** Consistency is key when it comes to building a daily drawing habit. Try to go to the same time and place every day, even on weekends.

Remember, building a daily drawing habit takes time and effort, but it's a rewarding hobby that can be very gratifying to see progress over time.
```

## References & Acknowledgements
- InstructGPT: [[arXiv](https://arxiv.org/abs/2203.02155)]
- DPO: [[arXiv](https://arxiv.org/abs/2305.18290)]
- OpenAssistant: [[GitHub](https://github.com/LAION-AI/Open-Assistant)] [[arXiv](https://arxiv.org/abs/2304.07327)]
- TRL - Transformer Reinforcement Learning: [[GitHub](https://github.com/huggingface/trl)]
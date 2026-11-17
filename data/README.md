## Data

| Stage | Dataset | Split |
|---|---|---|
| SFT | [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | `ready.trees` |
| Reward Model | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | `train_prefs` / `test_prefs` |
| PPO | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | `train_gen` (prompts only; same prompts as `train_prefs`) |
| DPO | [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | `train_prefs` / `test_prefs` |

### OASST2
```bash
bash scripts/get_data.sh
```
- Keeps English conversations only (`--language`)
- Keeps the highest-ranked reply at each turn (one conversation per tree)
- Prepends the system prompt `You are a helpful assistant.`
- Output: `data/oasst2/oasst2_messages.jsonl`

### UltraFeedback
Loaded from the Hugging Face Hub at training time.
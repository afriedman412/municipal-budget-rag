# VM Command Crib Sheet

## Start VM + vLLM
```bash
# Start VM and open iTerm tabs (local)
./start_vm.sh budget-mistral-lora-v7-merged

# Or manually SSH in and serve
gcloud compute ssh muni-rag-420 --zone=us-central1-a
cd municipal-budget-rag && source venv-vllm/bin/activate
vllm serve budget-mistral-lora-v7-merged --port 8000 --max-model-len 32768
```

## Run Test Suite (on VM, in venv-finetune)
```bash
python run_test_suite.py --model budget-mistral-lora-v7-merged --version v7 -w 8 --samples 5 --wandb
```

## Fine-tuning (on VM, in venv-finetune, use tmux)
```bash
tmux new -s train
python finetune.py \
  --data training/training_data_d4.jsonl \
  --epochs 5 \
  --lr 1e-4 \
  --lora-r 32 \
  --warmup 0.1 \
  --output budget-mistral-lora-v7
```

## Merge LoRA → Serveable Model (on VM, in venv-finetune)
```bash
python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained('budget-mistral-lora-v7')
model.save_pretrained_merged('budget-mistral-lora-v7-merged', tokenizer, save_method='merged_16bit')
"
```

## Copy Runs to Local
```bash
# From local machine
gcloud compute scp 'muni-rag-420:~/municipal-budget-rag/runs/*v7*' runs/ --zone=us-central1-a
```

## Stop VM (~$0.90/hr)
```bash
gcloud compute instances stop muni-rag-420 --zone=us-central1-a
```

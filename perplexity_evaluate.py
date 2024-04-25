#%%
import os
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
#%%
from datasets import load_dataset
from tqdm import tqdm
import gc
from perplexity import Perplexity
import argparse


model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=1024,
    use_fast=False,
    padding_side="right",
    truncation=True,
    max_length=1024,
    #padding=True
    # add_bos_token=False,
    # add_eos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

parser = argparse.ArgumentParser(description='Dataset generator for finetuning.')

parser.add_argument('--N', type=int, default=20_000,
                    help='Number of samples to use for training.')


parser.add_argument('--batch_size', type=int, default=10,
                    help='Number of samples to use for training.')

parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=1, help="Learning rate.")
parser.add_argument(
    "--prefix", type=str, default="fold-training", help="Prefix for model name."
)


args = parser.parse_args()

N = args.N
batch_size = args.batch_size

perplexity = Perplexity()

#%%
for shard_num in range(0, 5):
    print(f"Processing shard {shard_num}")
    shard_nums = []
    for j in range(0, 5):
        if j == shard_num: # make OOB shard
            continue
        shard_nums.append(str(j))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    ).half()

    model = PeftModel.from_pretrained(
        model,
        f"exps/{args.prefix}_epochs_{args.epochs}_lr_{args.lr}_R_128_ALPH_256_N_{N}_{'.'.join(shard_nums)}",
    )

    eval_data = load_dataset("json", data_files=f"shard_{N}_{shard_num}.jsonlines", split="train")

    shard_name = f"shard_{N}_{shard_num}_ppl.csv"

    shard_start = 0
    if os.path.exists(shard_name):
        with open(shard_name, "r") as f:
            shard_start = len(f.readlines())
    
    print(f"Starting at {shard_start} for shard {shard_num}")


    for idx in tqdm(range(shard_start, len(eval_data), 200)):
        start_idx = idx
        end_idx = idx + 200
        if end_idx > len(eval_data):
            end_idx = len(eval_data)

        chunk = eval_data[start_idx:end_idx]
        data = perplexity._compute(data=chunk["text"], batch_size=batch_size, model_id=model, tokenizer=tokenizer, max_length=1024)
        for ppl_idx in range(0, len(data["perplexities"])):
            ppl = data["perplexities"][ppl_idx]
            with open(shard_name, "a") as f:
                f.write(str(ppl) + "\n")

    del model
    torch.cuda.empty_cache()
    gc.collect()

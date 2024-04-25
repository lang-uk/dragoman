import torch
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
import transformers
import gc
import argparse
import wandb
import glob

os.environ["WANDB_PROJECT"] = "uk-translation-k-fold"

parser = argparse.ArgumentParser(description="Dataset generator for finetuning.")


# Required positional argument
parser.add_argument(
    "--N", type=int, default=20_000, help="Number of samples to use for training."
)
parser.add_argument(
    "--folds",
    nargs="+",
    type=int,
    default=[0, 1, 2, 3, 4],
    help="Folds to use for training.",
)
parser.add_argument(
    "--run_type",
    type=str,
    default="folds",
    help='Type of run. Types: "folds", "cleaned", "full".',
)
parser.add_argument(
    "--resume", type=bool, default=True, help="Continue training from checkpoint."
)
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=1, help="Learning rate.")
parser.add_argument(
    "--prefix", type=str, default="fold-training", help="Prefix for model name."
)
parser.add_argument(
    "--lora_checkpoint",
    type=str,
    default=None,
    help="Path to lora checkpoint to resume training.",
)

args = parser.parse_args()

PREFIX = args.prefix
MICRO_BATCH_SIZE = 8
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.epochs
LEARNING_RATE = args.lr
CUTOFF_LEN = 512
LORA_R = 128
LORA_ALPHA = 256
LORA_DROPOUT = 0.05
N = args.N
OUTPUT_MODEL_NAME = (
    f"{PREFIX}_epochs_{EPOCHS}_lr_{LEARNING_RATE}_R_{LORA_R}_ALPH_{LORA_ALPHA}_N_{N}"
)

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "mistralai/Mistral-7B-v0.1"


# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Preparing tokenized version according to the comment
# https://github.com/huggingface/transformers/issues/22794#issuecomment-1601482558


def tokenize(tokenizer, model_input_text: str, splitter: str = "[/INST] "):
    """Format and tokenize instruction tuning data

    1) Combine the user input (instruction) and agent response
    2) Create `labels` - ensuring we only fine tune over the
    desired agent response
    """
    orig, translated = model_input_text.split(splitter, 1)

    # Tokenize the full model input
    model_input = tokenizer(
        model_input_text, truncation=True, padding=False, return_tensors=None
    )

    # Create `labels` - ignoring user input (instructions)
    keep_tokens = tokenizer(translated).input_ids
    num_tokens_ignore = len(model_input["input_ids"]) - len(keep_tokens)
    model_input["num_tokens_ignore"] = [num_tokens_ignore]
    ignored_tokens = [-100] * num_tokens_ignore
    # Copy over the ids for the desired agent response
    model_input["labels"] = (
        ignored_tokens + model_input["input_ids"][-len(keep_tokens) :]
    )

    return model_input


def train_on_data(data, eval_data, run_info: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)

    model = prepare_model_for_kbit_training(model)

    if args.lora_checkpoint is not None:
        print("Resuming from existing lora checkpoint...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.lora_checkpoint,
            lora_config=LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                ],
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            is_trainable=True
        )
    else:
        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, config)


    data = data.map(lambda x: tokenize(tokenizer, x["text"]), num_proc=40)
    data = data.filter(lambda x: len(x["input_ids"]) <= CUTOFF_LEN)

    eval_enabled = False
    if eval_data is not None:
        eval_enabled = True
        eval_data = eval_data.map(lambda x: tokenize(tokenizer, x["text"]), num_proc=40)
        eval_data = eval_data.filter(lambda x: len(x["input_ids"]) <= CUTOFF_LEN)


    print("Dataset size after cutoff:", len(data))
    print("Max len:", max([len(x["input_ids"]) for x in data]))

    total_steps = int(
        (len(data) // (MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)) * EPOCHS
    )
    warmup_steps = min(100, int(total_steps * 0.1))
    print(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    wandb_run_name = (
        f"{OUTPUT_MODEL_NAME}_{run_info}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    with wandb.init(
        project=os.environ["WANDB_PROJECT"], reinit=True, name=wandb_run_name
    ) as run:
        # add script file to wandb
        artifact = wandb.Artifact(
            "finetune.py", type="code", description="finetune script"
        )
        artifact.add_file("finetune.py")
        run.log_artifact(artifact)

        output_dir = f"exps/{OUTPUT_MODEL_NAME}_{run_info}"
        resume = False
        if os.path.exists(output_dir):
            if len(glob.glob(os.path.join(output_dir, "checkpoint-*"))) > 0:
                resume = args.resume
                if resume:
                    print("Resuming from checkpoint")
        trainer = Trainer(
            model=model,
            train_dataset=data,
            eval_dataset=eval_data if eval_enabled else None,
            args=TrainingArguments(
                per_device_train_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                eval_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                eval_steps=20 if eval_enabled else None,
                per_device_eval_batch_size=2,
                num_train_epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                fp16=True,
                logging_steps=5,
                output_dir=output_dir,
                save_total_limit=2,
                save_strategy="steps",
                save_steps=20,
                report_to="wandb",
                run_name=wandb_run_name,
                do_eval=True if eval_enabled else False,
                evaluation_strategy="steps" if eval_enabled else "no",
                # lr_scheduler_type="cosine_with_restarts",
                # lr_scheduler_kwargs={"num_cycles": 3,},
                warmup_steps=warmup_steps,
            ),
            data_collator=DataCollatorForTokenClassification(
                tokenizer,
                pad_to_multiple_of=1,
            ),
        )
        model.config.use_cache = False
        trainer.train(resume_from_checkpoint=resume)

        model.save_pretrained(output_dir)


def main():
    if args.run_type == "folds":
        # select a fold to use as OOB
        for i in args.folds:
            print(f"Training on fold {i}")
            torch.cuda.empty_cache()
            gc.collect()
            data = []
            shard_nums = []
            for j in range(0, 5):
                if j == i:  # make OOB shard
                    continue
                shard_nums.append(str(j))
                data.append(
                    load_dataset(
                        "json", data_files=f"shard_{N}_{j}.jsonlines", split="train"
                    )
                )
            data = concatenate_datasets(data)
            eval_data = load_dataset(
                "json", data_files=f"shard_{N}_{i}.jsonlines", split="train"
            )
            print(f"Loaded dataset shards {','.join(shard_nums)}. Size: {len(data)}")

            train_on_data(data, eval_data, run_info=".".join(shard_nums))
    elif args.run_type == "cleaned":
        data = load_dataset(
            "json", data_files=f"shard_{N}_ppl_filtered.jsonlines", split="train"
        )
        print(f"Loaded cleaned dataset for {N}. Size: {len(data)}")
        eval_data = load_dataset("facebook/flores", "eng_Latn-ukr_Cyrl")["dev"]
        eval_data = eval_data.map(
            lambda x: {
                "text": f"[INST] {x['sentence_eng_Latn']} [/INST] {x['sentence_ukr_Cyrl']}"
            }
        )
        train_on_data(data, eval_data, run_info="cleaned")

    elif args.run_type == "full":
        data = []
        for i in range(0, 5):
            data.append(
                load_dataset(
                    "json", data_files=f"shard_{N}_{i}.jsonlines", split="train"
                )
            )
        data = concatenate_datasets(data)
        print(f"Loaded full dataset for {N}. Size: {len(data)}")
        eval_data = load_dataset("facebook/flores", "eng_Latn-ukr_Cyrl")["dev"]
        eval_data = eval_data.map(
            lambda x: {
                "text": f"[INST] {x['sentence_eng_Latn']} [/INST] {x['sentence_ukr_Cyrl']}"
            }
        )
        train_on_data(data, eval_data, run_info="full")
    else:
        raise Exception("Invalid run type")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

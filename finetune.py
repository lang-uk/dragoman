import torch
from datetime import datetime
from datasets import load_dataset
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


os.environ["WANDB_PROJECT"] = "finetune_experiments"

MICRO_BATCH_SIZE = 8
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1
LEARNING_RATE = 2e-5
CUTOFF_LEN = 512
LORA_R = 256
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
OUTPUT_MODEL_NAME = "towerbase-translate-uk-0.19.full-lora.4bit.diff-tokenizer.sophiag.3m_filtered"
USE_SOPHIA_G = True

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "huggyllama/llama-7b"
# model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "upstage/SOLAR-10.7B-v1.0"
model_name = "Unbabel/TowerBase-7B-v0.1"


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


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"exps/{OUTPUT_MODEL_NAME}")

    data = load_dataset(
        "json", data_files="./data/processed/paracrawl_3m.jsonlines", split="train"
    )

    data = data.map(
        lambda x: tokenize(tokenizer, x["text"]), num_proc=40
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

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

    original_size = len(data)
    print(f"Source data size: {original_size}")

    training_args = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        output_dir=f"exps/{OUTPUT_MODEL_NAME}",
        save_total_limit=5,
        save_strategy="steps",
        save_steps=50,
        report_to="wandb",
        run_name=f"{OUTPUT_MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    if USE_SOPHIA_G:
        from optimizers.sophia import SophiaG

        optimizer = SophiaG(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
        )
    else:
        optimizer = None

    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=DataCollatorForTokenClassification(
            tokenizer,
            pad_to_multiple_of=1,
        ),
        optimizers=(optimizer, None),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=True)

    model.save_pretrained(f"exps/{OUTPUT_MODEL_NAME}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

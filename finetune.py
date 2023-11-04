import torch
from datetime import datetime
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os



os.environ["WANDB_PROJECT"]="finetune_experiments"

MICRO_BATCH_SIZE = 32
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 5e-5
CUTOFF_LEN = 512  # 1024 accounts for about 99.5% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
OUTPUT_MODEL_NAME = "mistral-instruct-translate-uk-0.05.full-lora.4bit.diff-tokenizer"


# peft_parameters = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM"
# )


model_name = "mistralai/Mistral-7B-Instruct-v0.1"


# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="left",
        add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

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

    data = load_dataset("json", data_files="/tmp/paracrawl.jsonlines", split="train")

    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            # truncation=True,
            # max_length=CUTOFF_LEN,
            #padding=True#"max_length",
        )
        return result


    data = data.shuffle().map(lambda x: tokenize(x["text"]), num_proc=40)

    original_size = len(data)
    print(f"Source data size: {original_size}")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=100,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=50,
            output_dir=f"exps/{OUTPUT_MODEL_NAME}",
            save_total_limit=15,
            save_strategy="steps",
            save_steps=50,
            report_to="wandb",
            run_name=f"{OUTPUT_MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=1),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)


    model.save_pretrained(f"exps/{OUTPUT_MODEL_NAME}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

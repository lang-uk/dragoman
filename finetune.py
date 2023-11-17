import torch
from typing import Any, List, Dict, Union, Mapping
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.data.data_collator import _torch_collate_batch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os


os.environ["WANDB_PROJECT"] = "finetune_experiments"

MICRO_BATCH_SIZE = 32
BATCH_SIZE = 256
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  # we don't need 3 tbh
LEARNING_RATE = 5e-5
CUTOFF_LEN = 512  # 1024 accounts for about 99.5% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
OUTPUT_MODEL_NAME = "mistral-translate-uk-0.07.full-lora.4bit.diff-tokenizer"

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

    # # Just to demonstrate length equality
    # assert len(model_input["labels"]) == len(model_input["input_ids"])

    return model_input


class RiggedDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            print(examples)
            batch = self.tokenizer.pad(
                examples,
                padding=True,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }

        print("Here is johynn")
        print(batch)
        # labels = batch["input_ids"].clone()
        # if self.tokenizer.pad_token_id is not None:
        #     labels[labels == self.tokenizer.pad_token_id] = -100
        # batch["labels"] += [-100] * (len(batch["input_ids"]) - len(batch["labels"]))

        # ignored_tokens = [-100] * (batch["num_tokens_ignore"])
        # # # Copy over the ids for the desired agent response
        # batch["labels"] = (
        #     ignored_tokens + labels[batch["num_tokens_ignore"]:]
        # )

        assert len(batch["labels"]) == len(batch["input_ids"])
        # raise Exception("Fuck you")
        return batch


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=False,
        # padding=True,
        # model_input_names=["input_ids", "token_type_ids", "attention_mask", "labels"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(f"exps/{OUTPUT_MODEL_NAME}")
    # tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    data = load_dataset("json", data_files="/tmp/paracrawl.jsonlines", split="train")

    data = data.shuffle().map(lambda x: tokenize(tokenizer, x["text"]), num_proc=40)
    # print(data[0])


    # data = data.remove_columns(["text", "num_tokens_ignore"])

    # print(tokenizer.pad(data, return_tensors="pt", pad_to_multiple_of=1,))
    # collator = DataCollatorForTokenClassification(
    #     tokenizer, pad_to_multiple_of=1
    # )
    # from torch.utils.data import DataLoader
    # dl = DataLoader(data, batch_size=10, collate_fn=collator)

    # for batch in dl:
    #     print(batch)
    #     break

    # return

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

    trainer = Trainer(
        model=model,
        train_dataset=data,
        args=TrainingArguments(
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
            run_name=f"{OUTPUT_MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        ),
        data_collator=DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=1,
        ),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    model.save_pretrained(f"exps/{OUTPUT_MODEL_NAME}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

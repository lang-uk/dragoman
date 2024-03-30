import argparse
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

import wandb
from unsloth import FastLanguageModel

# Move me somewhere else
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",
    "unsloth/gemma-7b-it-bnb-4bit",  # Instruct version of Gemma 7b
    "unsloth/gemma-2b-bnb-4bit",
    "unsloth/gemma-2b-it-bnb-4bit",  # Instruct version of Gemma 2b
    "unsloth/solar-10.7b-bnb-4bit",
]  # More models at https://huggingface.co/unsloth


from decode import BatchTranslator, Prompter


# Preparing tokenized version according to the comment
# https://github.com/huggingface/transformers/issues/22794#issuecomment-1601482558


def tokenize(tokenizer, model_input_text: str, sep: str = "[/INST] "):
    """Format and tokenize instruction tuning data

    1) Combine the user input (instruction) and agent response
    2) Create `labels` - ensuring we only fine tune over the
    desired agent response
    """
    orig, translated = model_input_text.split(sep, 1)

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


def main(args):
    wandb.init(
        project="finetune_experiments",
        config=vars(args),
        name=f"{args.exp}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    )

    data = load_dataset(
        "json",
        data_files=args.train,
        split="train",
    )

    print("Loading data from:", args.train + ", found", len(data), "examples")
    print("First training example:", data[0])
    print("Using separator for conditional LM training:", prompter.separator)

    print(args.model_name_or_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name_or_path,
        max_seq_length=args.model_max_length,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(args.exp)

    data = data.map(
        lambda x: tokenize(tokenizer, x["text"], sep=prompter.separator),
        num_proc=40,
        desc="Tokenizing",
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,  # Supports any, but = 0 is optimized
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",],
    #     lora_alpha = 16,
    #     lora_dropout = 0, # Supports any, but = 0 is optimized
    #     bias = "none",    # Supports any, but = "none" is optimized
    #     use_gradient_checkpointing = True,
    #     random_state = 3407,
    #     use_rslora = False,  # We support rank stabilized LoRA
    #     loftq_config = None, # And LoftQ
    # )
    # if args.optimizer == "sophiag":
    #     from optimizers.sophia import SophiaG

    #     optimizer = SophiaG(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=args.learning_rate,
    #     )
    # else:
    #     optimizer = None

    # TODO: collator
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data,
        dataset_text_field="text",
        max_seq_length=args.model_max_length,
        dataset_num_proc=40,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=1,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            output_dir=args.exp,
            save_total_limit=args.save_total_limit,
            save_strategy="steps",
            save_steps=args.save_steps,
            report_to="wandb",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
        ),
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)

    if args.save_total_limit > 0:
        model.save_pretrained(args.exp)

    # if args.decode_beams:
    #     print('Decoding FLORES', args.decode_subset)
    #     model = model.merge_and_unload()
    #     # TODO: maybe convert the whole thing to float16?
    #     model.gradient_checkpointing_disable()
    #     translator = BatchTranslator(
    #         decode_beams=args.decode_beams,
    #         decode_batch_size=args.decode_batch_size,
    #         model=model,
    #         tokenizer=BatchTranslator.load_tokenizer(BatchTranslator.get_base_model(args)),
    #         prompter=prompter
    #     )
    #     results = translator.decode_flores(exp=args.exp, decode_subset=args.decode_subset)
    #     wandb.log({
    #         'decode/bleu': results['score'],
    #         'decode/ref_len': results['ref_len'],
    #         'decode/hyp_len': results['sys_len'],
    #     })


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Unsloth train loop", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train",
        default="data/processed/paracrawl_filtered_alpaca.jsonlines",
        type=str,
        help="A jsonlines file containing the training data.",
    )
    # parser.add_argument(
    #     "--optimizer",
    #     default="adamw",
    #     choices=["sophiag", "adamw"],
    #     type=str,
    #     help="Optimizer.",
    # )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=4,
        type=int,
        help="Batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=64,
        type=int,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="Learning rate."
    )
    parser.add_argument("--lora_rank", default=256, type=int, help="LoRA adapter rank.")
    parser.add_argument("--lora_alpha", default=512, type=int, help="LoRA alpha.")
    parser.add_argument(
        "--lora_dropout",
        default=0.0,
        type=float,
        help="LoRA dropout (using 0 for unsloth).",
    )
    parser.add_argument(
        "--model_max_length", default=2048, type=int, help="Maximum model input length."
    )
    parser.add_argument(
        "--save_steps", default=50, type=int, help="Save checkpoints every X steps."
    )
    parser.add_argument(
        "--save_total_limit",
        default=5,
        type=int,
        help="Limit the total amount of checkpoints.",
    )

    BatchTranslator.register(parser)  # --exp, --prompt are here
    args = parser.parse_args()
    prompter = Prompter(args.prompt)

    main(args=args)

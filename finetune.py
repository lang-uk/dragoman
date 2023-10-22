import torch
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


MICRO_BATCH_SIZE = 32  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 512  # 1024 accounts for about 99.5% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# peft_parameters = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM"
# )


model_name = "mistralai/Mistral-7B-v0.1"


# data_name = "mlabonne/guanaco-llama2-1k"
# training_data = load_dataset(data_name, split="train")

# # Model and tokenizer names
# base_model_name = "NousResearch/Llama-2-7b-chat-hf"
# refined_model = "llama-2-7b-mlabonne-enhanced" #You can give it your own name

# # Tokenizer
# llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
# llama_tokenizer.pad_token = llama_tokenizer.eos_token
# llama_tokenizer.padding_side = "right"  # Fix for fp16

# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# # Model
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=quant_config,
#     device_map={"": 0}
# )
# base_model.config.use_cache = False
# base_model.config.pretraining_tp = 1

def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        # load_in_8bit=True,
        device_map="auto",
        # device_map={'': torch.cuda.current_device()}
        # device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, add_eos_token=True
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT, 
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

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
            logging_steps=100,
            output_dir="exps/mistral-translate-uk-0.01",
            save_total_limit=3,
            save_strategy="epoch",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=1),
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=False)


    model.save_pretrained("exps/mistral-translate-uk-0.02.4bit")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

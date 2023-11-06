import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)

model_name = "mistralai/Mistral-7B-v0.1"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=1024,
    use_fast=False,
    padding_side="left",
    # add_eos_token=False,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
).half()

model = PeftModel.from_pretrained(
    model,
    "exps/mistral-translate-uk-0.06.full-lora.4bit.diff-tokenizer/checkpoint-1600/",
)


generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
    do_sample=True,
)


def generate_prompt(instruction, input=None) -> str:
    return f"[INST] {instruction} [/INST] "


def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256,
        use_cache=False,
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Відповідь:", output)


instruction = "Hello team! How are you today?"
print("Запит:", instruction)
evaluate(instruction)

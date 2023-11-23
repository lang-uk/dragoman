import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

torch.set_default_device('cpu')

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    model_max_length=1024,
    use_fast=False,
    padding_side="right",
    # add_bos_token=False,
    # add_eos_token=False,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cpu"},
).to("cpu")

model = PeftModel.from_pretrained(
    model,
    "exps/mistral-translate-uk-0.07.full-lora.4bit.diff-tokenizer/checkpoint-3750",
    device_map={"": "cpu"},
).to("cpu")

generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
    do_sample=True,
)


def generate_prompt(instruction, input=None) -> str:
    return f"[INST] {instruction} [/INST]"


def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    input_ids = inputs["input_ids"]
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

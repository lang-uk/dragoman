from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "exps/mistral-translate-uk-0.01/checkpoint-7524/")


generation_config = GenerationConfig(
    temperature=0.2,
    top_p=0.75,
    num_beams=4,
)

def generate_prompt(instruction, input=None)-> str:
    return f"<s>[INST] {instruction} [/INST] "

def evaluate(instruction, input=None):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
        print("Відповідь:", output)



instruction = "Ground control to major Tom Commencing countdown, engines on."
print("Запит:", instruction)
evaluate(instruction)
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, "exps/mistral-translate-uk-0.01/checkpoint-1504/")


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



instruction = "Franks grew up in southern California with his father Thurman, his mother Vera, and two younger sisters. Although no one in his family was a musician, his parents loved swing music, and his early influences included Peggy Lee, Nat King Cole, Ira Gershwin, Irving Berlin, and Johnny Mercer. At age 14 Franks bought his first guitar, a Japanese Marco Polo for $29.95 with six private lessons included; those lessons were the only music education that he received."
print("Запит:", instruction)
evaluate(instruction)
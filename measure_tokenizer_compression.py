import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize(tokenizer, model_input_text: str, drop_orig=False):
    if drop_orig:
        _, model_input_text = model_input_text.split("[/INST] ", 1)

    model_input = tokenizer(
        model_input_text, truncation=True, padding=False, return_tensors=None
    )
    return {"tokens": len(model_input["input_ids"]), "characters": len(model_input_text)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    parser.add_argument("input_file", type=str, help="Input file")
    parser.add_argument("--drop_orig", default=False, action="store_true", help="Drop original text")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=False,
    )

    data = load_dataset(
        "json", data_files=args.input_file, split="train"
    )

    data = data.map(lambda x: tokenize(tokenizer, x["text"], drop_orig=args.drop_orig), num_proc=40)

    texts = 0
    characters = 0
    tokens = 0

    for d in data:
        texts += 1
        characters += d["characters"]
        tokens += d["tokens"]
    
    print(f"Evaluating {args.model_name} tokenizer compression")
    print(f"Texts: {texts}, Characters: {characters}, Tokens: {tokens}")

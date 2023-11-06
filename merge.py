import argparse
from transformers import AutoModelForCausalLM
from peft import PeftModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("checkpoint")
    parser.add_argument("output_path")

    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(args.output_path)

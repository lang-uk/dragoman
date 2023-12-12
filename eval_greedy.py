from typing import Dict, List
import os.path
import logging
import argparse
import csv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def generate_prompt(instruction: str, input=None) -> str:
    return f"[INST] {instruction} [/INST]"


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--checkpoints", nargs="*")
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--dataset", default="data/flores_eng_ukr_major.csv")
    parser.add_argument("--preset", default="greedy", choices=["greedy"])

    args = parser.parse_args()

    logger.info(f"Got {len(args.checkpoints)} checkpoints to evaluate")

    logger.info(f"Loading tokenizer {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=1024,
        use_fast=False,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset {args.dataset}")
    dataset: List[Dict] = []
    with open(args.dataset, "r", encoding="utf8") as fp_in:
        reader = csv.DictReader(fp_in)
        for d in reader:
            prompt = generate_prompt(d["sentence_eng_Latn"], input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()

            dataset.append(
                {
                    "id": d["id"],
                    "orig": d["sentence_eng_Latn"],
                    "trans": d["sentence_ukr_Cyrl"],
                    "prompt": prompt,
                    "input_ids": input_ids,
                }
            )

    logger.info(f"Loaded and tokenized {len(dataset)} examples")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
    ).half()

    logger.info("Loaded base model")

    for checkpoint in tqdm(args.checkpoints):
        logger.info(f"Loading checkpoint {checkpoint}")
        checkpoint_slug = checkpoint.replace("/", "-")
        output_file = f"{args.output_dir}/{checkpoint_slug}.csv"

        if os.path.exists(output_file):
            logger.info(f"Skipping {checkpoint} - already exists")
            continue

        peft_model = PeftModel.from_pretrained(
            model,
            checkpoint,
        )

        with open(output_file, "w", encoding="utf8") as fp_out:
            w = csv.DictWriter(fp_out, fieldnames=["id", "orig", "reference", "generated"])
            w.writeheader()

            for example in dataset:
                generation_output = peft_model.generate(
                    input_ids=example["input_ids"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=256,
                    use_cache=False,
                )

                for s in generation_output.sequences:
                    output = tokenizer.decode(s)
                    if "[/INST]" in output:
                        _, output = output.split("[/INST]", 1)
                        output = output.replace("<s>", "").replace("</s>", "").strip()
                    else: # No response
                        logger.warning(f"Got invalid response for {example['id']}: {output}")
                        output = ""

                    print("Відповідь:", output)
                    w.writerow(
                        {
                            "id": example["id"],
                            "orig": example["orig"],
                            "reference": example["trans"],
                            "generated": output,
                        }
                    )
            break

import argparse
import json
from datasets import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=argparse.FileType('w'))
    parser.add_argument("--direction", type=str, default="uk-en", choices=["uk-en", "en-uk"])
    args = parser.parse_args()

    dataset = load_dataset("turuta/Multi30k-uk")

    for example in dataset["train"]:
        if args.direction == "uk-en":
            text = f"[INST] {example['uk']} [/INST] {example['en']}"
        else:
            text = f"[INST] {example['en']} [/INST] {example['uk']}"

        args.output_file.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

    args.output_file.close()

"""
Convert pararook sentences from export_pararook.py to a set of instructions
suitable for training.
"""

import json
import argparse
from tqdm import tqdm


def main(args) -> None:
    """
    Main function to parse JSONL file and write instructions to a new JSONL file.

    Args:
        args: Parsed command-line arguments.
    """
    source_lang, target_lang = args.lang_pair.split("-")

    for line in tqdm(args.input_file):
        data = json.loads(line)
        instruction = f"[INST] {data[source_lang]} [/INST] {data[target_lang]}"
        if len(instruction) <= args.max_length:
            args.output_file.write(json.dumps({"text": instruction}, ensure_ascii=False) + "\n")

    args.input_file.close()
    args.output_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export sentences of parallel corpora into a set of instructions."
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input file in JSONL format"
    )
    parser.add_argument(
        "lang_pair",
        type=str,
        choices=["en-uk", "uk-en"],
    )
    parser.add_argument(
        "--max-length", 
        type=int,
        default=1024,
        help="Skip instructions longer than this length"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="Output file in JSONL format"
    )

    main(parser.parse_args())

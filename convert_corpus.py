import json
from collections import Counter
import argparse
import pathlib

import smart_open

def main(args):
    lengths = Counter()
    with smart_open.open(args.input_path, "r", encoding="utf-8") as fp_in:
        with smart_open.open(args.output_path, "w", encoding="utf-8") as fp_out:
            for line_no, (orig, trans) in enumerate(map(lambda x: x.split("\t", 1), fp_in)):
                if line_no >= args.first_n:
                    break
                
                text = args.template.format(orig=orig.strip(), translated=trans.strip())
                if len(text) <= args.filter_by_len:
                    fp_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                lengths.update([len(text)])
    
    # print(lengths.most_common())
    running_total = 0
    full_total = sum(lengths.values()) or 1
    target_lengths = {256: False, 512: False, 768: False, 1024: False, 2048: False}

    for k in sorted(lengths.keys()):
        running_total += lengths[k]
        
        for target, flag in target_lengths.items():
            if k >= target and not flag:
                print(k, running_total / full_total)
                target_lengths[target] = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=pathlib.Path, help="Path to the file with the parallel corpus in tab separated format"
    )
    parser.add_argument(
        "output_path",
        type=pathlib.Path,
        help="Path to the jsonl file with generated instructions for lora finetuning",
    )
    parser.add_argument(
        "--template",
        default="[INST] {orig} [/INST] {translated}",
        help="Instruction template"
    )
    parser.add_argument(
        "--first-n",
        type=int,
        default=1000,
        help="Number of sentences to draw from the input file"
    )
    parser.add_argument(
        "--filter-by-len",
        type=int,
        default=1000000,
        help="Do not include instructions longer than that"
    )
    args = parser.parse_args()

    main(args=args)

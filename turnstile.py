import re
import json
import argparse
from tqdm import tqdm

switch_regex = re.compile(r"^\[INST\](.*)\[\/INST\](.*)$", re.MULTILINE)

def switch(args):
    for l in tqdm(args.input.readlines()):
        match = switch_regex.match(json.loads(l)["text"])
        if match:
            args.output.write(
                json.dumps({
                    "text": f"[INST]{match.group(2).strip()} [/INST]{match.group(1).strip()}",
                }, ensure_ascii=False) + "\n"
            )
        else:
            print(f"Could not switch text in line: {l}")

    args.input.close()
    args.output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command line tool for managing and manipulating parallel corpus")

    parser.add_argument("input", help="The file to switch the text in", type=argparse.FileType("r"))
    parser.add_argument("output", help="The file to write the output to", type=argparse.FileType("w"))

    args = parser.parse_args()
    switch(args)

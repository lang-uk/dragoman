import argparse
import json
from pathlib import Path
from tqdm import tqdm


def generate_wordnet_prompt(instruction: str) -> str:
    return (
        f"Ти професор української лінгвістики, лексикограф, автор і укладач словників. "
        + f"Нижче наведено задачу, разом з вхідними даними до неї. Надай найкращу відповідь "
        + f" ### Instruction: {instruction} ### Response: "
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
    )
    parser.add_argument(
        "output",
        type=Path,
    )
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as fp:
        with args.output.open("w", encoding="utf-8") as fp_out:
            for line in tqdm(fp):
                task_data = json.loads(line)
                instruction = generate_wordnet_prompt(task_data["instruction"])
                text = instruction + " " + task_data["output"] + "."

                fp_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

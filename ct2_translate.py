import argparse
import pathlib
from tokenizers import Tokenizer
import ctranslate2


def translate(args):
    generator = ctranslate2.Generator(
        str(args.model),
        device="auto",
    )

    tokenizer = Tokenizer.from_file(str(args.tokenizer))

    while True:
        prompt = input("Text to translate: ")
        prompt = f"[INST] {prompt} [/INST] "

        tokens = tokenizer.encode(prompt).tokens
        print(tokens)

        results = generator.generate_batch(
            [tokens],
            beam_size=5,
            include_prompt_in_result=False,
            repetition_penalty=1.05,
        )
        prediction = tokenizer.decode(results[0].sequences_ids[0])
        print(prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=pathlib.Path,
        help="Path to the merged and converted ctranslate model",
    )
    parser.add_argument(
        "tokenizer",
        type=pathlib.Path,
        help="Path to the tokenizer config from the original model",
    )

    args = parser.parse_args()

    translate(args=args)

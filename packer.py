import csv
import json
import random
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
from enum import StrEnum


class AlgoEnum(StrEnum):
    PACK_MAX = "pack_max"
    RANDOM_20 = "random_20"

def preprocess_dataset(
    input_file: str,
    output_file: str,
    token_limit: int,
    model_name: str,
    algo: AlgoEnum = AlgoEnum.PACK_MAX,
    separator: str = ", ",
) -> None:
    """
    Preprocesses the dataset by grouping consecutive records into instructions under the specified token limit.

    Args:
    input_file (str): Path to the input CSV file.
    output_file (str): Path to the output jsonlines file to save the preprocessed data.
    token_limit (int): Maximum token limit for each instruction.
    model_name (str): Name of the pretrained model.

    Returns:
    None
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=1024,
        use_fast=False,
        padding_side="right",
        add_eos_token=False,
        add_bos_token=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    inst_token_count = len(tokenizer("[INST]").input_ids)
    end_inst_token_count = len(tokenizer("[/INST]").input_ids)
    comma_token_count = len(tokenizer(separator.strip()).input_ids)

    current_instruction_tokens = (
        inst_token_count + end_inst_token_count - 2 * comma_token_count
    )

    current_instruction_translits = []
    current_instruction_names = []

    def instruction_template(translits: List[str], names: List[str]) -> str:
        return f"[INST]{separator.join(translits)}[/INST] {separator.join(names)}"

    def write_instruction(fp, translits: List[str], names: List[str]) -> None:
        fp.write(
            json.dumps(
                {"instruction": instruction_template(translits, names)},
                ensure_ascii=False,
            )
            + "\n"
        )

    # Load data and shuffle
    with open(input_file, encoding="utf-8") as csvfile:
        reader = list(csv.DictReader(csvfile))
        random.seed(42)
        random.shuffle(reader)

        next_batch = 1000 # Whatever, any big number in fact
        if algo == AlgoEnum.RANDOM_20:
            next_batch = random.randrange(1, 21)

        with open(output_file, "w", encoding="utf-8") as fp_out:
            for row in tqdm(reader):
                # Calculate token count for the current row
                row_tokens = (
                    len(tokenizer(row["name"]).input_ids)
                    + len(tokenizer(row["translit"]).input_ids)
                    + comma_token_count * 2
                )

                start_new = current_instruction_tokens + row_tokens > token_limit
                if algo == AlgoEnum.RANDOM_20:
                    start_new = start_new or len(current_instruction_translits) >= next_batch

                # If adding the current row exceeds the token limit, start a new instruction
                if start_new:
                    write_instruction(
                        fp_out, current_instruction_translits, current_instruction_names
                    )

                    # Reset current instruction
                    current_instruction_tokens = (
                        inst_token_count + end_inst_token_count - 2 * comma_token_count
                    )
                    current_instruction_translits = []
                    current_instruction_names = []

                    if algo == AlgoEnum.RANDOM_20:
                        next_batch = random.randrange(1, 21)

                # Add row to current instruction
                current_instruction_translits.append(row["translit"])
                current_instruction_names.append(row["name"])
                current_instruction_tokens += row_tokens

            # Append the last instruction
            if current_instruction_translits:
                write_instruction(
                    fp_out, current_instruction_translits, current_instruction_names
                )

    # # Calculate and verify total token count
    # total_tokens = sum(
    #     tokenizer(inst["instruction"], return_tensors="pt").input_ids.numel()
    #     for inst in instructions
    # )
    # assert (
    #     total_tokens <= token_limit
    # ), f"Total tokens ({total_tokens}) exceed token limit ({token_limit})"


if __name__ == "__main__":
    preprocess_dataset(
        input_file="data/translit_names_filtered.csv",
        output_file="preprocessed_dataset_mistral_random.jsonl",
        token_limit=500,
        algo=AlgoEnum.RANDOM_20,
        # model_name="Unbabel/TowerBase-7B-v0.1",
        model_name="mistralai/Mistral-7B-v0.1",
    )
